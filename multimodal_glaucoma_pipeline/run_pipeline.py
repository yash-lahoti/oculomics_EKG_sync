#!/usr/bin/env python3
"""
Multimodal Glaucoma AI Pipeline

Unified analysis pipeline for:
- OCTA Vessel Density (ETDRS sectors + global; optionally slab-level)
- Fundus C/D (VCDR, area CDR, rim metrics, quality/device)
- RNFL thickness maps (global/quadrants/clock-hours + optionally grid)
- Clinical parameters (age, sex, IOP, A1c, BP, diagnosis, VF MD, etc.)

Design principle: everything becomes a "long-form feature table" keyed by
patient/eye/visit/modality/feature/value, then the same analysis recipes
(descriptives, LMM, AUROC, reliability) run regardless of modality.
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a table from CSV or Excel file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def melt_wide(
    df: pd.DataFrame, id_cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    """Melt a wide dataframe to long format."""
    # Only include id_cols that exist in the dataframe
    valid_id_cols = [c for c in id_cols if c in df.columns]
    valid_value_cols = [c for c in value_cols if c in df.columns]

    out = df.melt(
        id_vars=valid_id_cols,
        value_vars=valid_value_cols,
        var_name="feature",
        value_name="value",
    )
    return out


def select_cols_by_regex(df: pd.DataFrame, regex: str) -> list[str]:
    """Select columns matching a regex pattern."""
    pat = re.compile(regex)
    return [c for c in df.columns if pat.search(c)]


def format_p(p: float) -> str:
    """Format p-value for display."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def categorize_feature(full_feature: str, rules: list[dict]) -> str:
    """Assign a category to a feature based on regex rules."""
    for r in rules:
        if re.search(r["match"], full_feature):
            return r["category"]
    return "Other"


def safe_mean(series: pd.Series) -> float:
    """Calculate mean, returning NaN for empty series."""
    if series.empty or series.isna().all():
        return np.nan
    return float(np.nanmean(series))


def safe_std(series: pd.Series) -> float:
    """Calculate std, returning NaN for empty or single-value series."""
    if series.empty or series.isna().all() or series.notna().sum() < 2:
        return np.nan
    return float(np.nanstd(series, ddof=1))


# -----------------------------------------------------------------------------
# Statistical Analysis Functions
# -----------------------------------------------------------------------------


def run_lmm(
    df: pd.DataFrame, formula: str, group_col: str
) -> Optional[Any]:
    """
    Run a linear mixed model using statsmodels.

    Args:
        df: Input dataframe
        formula: R-style formula (e.g., "value ~ group + age + sex")
        group_col: Column to use for random effects grouping

    Returns:
        Fitted model or None if fitting fails
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.warning("statsmodels not installed; skipping LMM analysis")
        return None

    # Drop rows with missing values in key columns
    dd = df.dropna(subset=["value"]).copy()

    if dd.empty or dd[group_col].nunique() < 2:
        return None

    try:
        model = smf.mixedlm(formula, dd, groups=dd[group_col]).fit(reml=False)
        return model
    except Exception as e:
        logger.debug(f"LMM fitting failed: {e}")
        return None


def run_auroc(
    df: pd.DataFrame, label_col: str, pos_label: str
) -> float:
    """
    Calculate AUROC using logistic regression.

    Args:
        df: Input dataframe with 'value' column
        label_col: Column containing group labels
        pos_label: Value indicating positive class

    Returns:
        AUROC score or NaN if calculation fails
    """
    dd = df.dropna(subset=["value", label_col]).copy()

    if dd.empty:
        return np.nan

    y = (dd[label_col].astype(str) == str(pos_label)).astype(int).values
    x = dd["value"].values.reshape(-1, 1)

    # Check for degenerate cases
    if len(np.unique(y)) < 2:
        return np.nan

    if np.isnan(x).any():
        return np.nan

    try:
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(x, y)
        p = clf.predict_proba(x)[:, 1]
        return float(roc_auc_score(y, p))
    except Exception as e:
        logger.debug(f"AUROC calculation failed: {e}")
        return np.nan


def calculate_icc(
    df: pd.DataFrame, value_col: str, subject_col: str, rater_col: str
) -> Optional[float]:
    """
    Calculate Intraclass Correlation Coefficient (ICC) for reliability analysis.

    Uses ICC(2,1) - two-way random effects, single measurement, absolute agreement.

    Args:
        df: Input dataframe
        value_col: Column containing measurements
        subject_col: Column identifying subjects
        rater_col: Column identifying raters/devices

    Returns:
        ICC value or None if calculation fails
    """
    try:
        import pingouin as pg

        icc_results = pg.intraclass_corr(
            data=df,
            targets=subject_col,
            raters=rater_col,
            ratings=value_col,
        )
        # Return ICC(2,1)
        icc21 = icc_results[icc_results["Type"] == "ICC2"]["ICC"].values
        if len(icc21) > 0:
            return float(icc21[0])
    except ImportError:
        logger.debug("pingouin not installed; skipping ICC calculation")
    except Exception as e:
        logger.debug(f"ICC calculation failed: {e}")

    return None


# -----------------------------------------------------------------------------
# Main Pipeline Class
# -----------------------------------------------------------------------------


class MultiModalPipeline:
    """
    Unified pipeline for multimodal glaucoma analysis.

    Handles data loading, QC, merging, and statistical analysis across
    OCTA, Fundus, RNFL, and clinical modalities.
    """

    def __init__(self, cfg: dict, base_dir: Optional[Path] = None):
        """
        Initialize the pipeline.

        Args:
            cfg: Configuration dictionary loaded from YAML
            base_dir: Base directory for resolving relative paths
        """
        self.cfg = cfg
        self.base_dir = base_dir or Path.cwd()
        self.out_dir = self.base_dir / Path(cfg["project"]["output_dir"])

        # Create output directories
        ensure_dir(self.out_dir)
        ensure_dir(self.out_dir / "tables")
        ensure_dir(self.out_dir / "models")
        ensure_dir(self.out_dir / "qc")

        # Key column names
        self.k_patient = cfg["keys"]["patient_id"]
        self.k_eye = cfg["keys"]["eye"]
        self.k_visit = cfg["keys"]["visit_date"]

        # Set random seed
        np.random.seed(cfg["project"].get("random_seed", 42))

        logger.info(f"Initialized pipeline: {cfg['project']['name']}")
        logger.info(f"Output directory: {self.out_dir}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def load_clinical(self) -> pd.DataFrame:
        """Load and preprocess clinical data."""
        path = self._resolve_path(self.cfg["clinical"]["path"])
        logger.info(f"Loading clinical data from: {path}")

        clin = read_table(path)

        # Standardize key column types
        for k in [self.k_patient, self.k_eye, self.k_visit]:
            if k in clin.columns:
                clin[k] = clin[k].astype(str)

        logger.info(f"Loaded {len(clin)} clinical records")
        return clin

    def modality_to_long(self, mod_cfg: dict) -> pd.DataFrame:
        """
        Convert a modality's data to long format.

        Args:
            mod_cfg: Modality configuration from config.yaml

        Returns:
            Long-format dataframe with columns:
            [id_cols..., modality, feature, feature_full, value]
        """
        path = self._resolve_path(mod_cfg["path"])
        logger.info(f"Loading {mod_cfg['name']} data from: {path}")

        df = read_table(path).copy()

        # Standardize key column types
        for k in [self.k_patient, self.k_eye, self.k_visit]:
            if k in df.columns:
                df[k] = df[k].astype(str)

        id_cols = mod_cfg["id_cols"]

        # Convert to long format
        if mod_cfg["format"] == "long":
            feat_col = mod_cfg.get("long_feature_col", "feature")
            val_col = mod_cfg.get("long_value_col", "value")
            cols_to_keep = [c for c in id_cols + [feat_col, val_col] if c in df.columns]
            out = df[cols_to_keep].rename(
                columns={feat_col: "feature", val_col: "value"}
            )
        else:
            # Wide format - melt to long
            if "value_cols" in mod_cfg:
                value_cols = mod_cfg["value_cols"]
            else:
                value_cols = select_cols_by_regex(df, mod_cfg["value_cols_regex"])

            if not value_cols:
                logger.warning(f"No value columns found for {mod_cfg['name']}")
                return pd.DataFrame()

            out = melt_wide(df, id_cols=id_cols, value_cols=value_cols)

        # Add modality identifier and namespace
        ns = mod_cfg.get("feature_namespace", mod_cfg["name"]).upper()
        out["modality"] = mod_cfg["name"]
        out["feature"] = out["feature"].astype(str)
        out["feature_full"] = ns + ":" + out["feature"]

        # Ensure numeric values
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

        logger.info(f"Converted {mod_cfg['name']} to long format: {len(out)} records")
        return out

    def apply_qc(self, long_df: pd.DataFrame, mod_cfg: dict) -> pd.DataFrame:
        """
        Apply quality control filters to modality data.

        Args:
            long_df: Long-format data
            mod_cfg: Modality configuration

        Returns:
            Filtered dataframe
        """
        if long_df.empty:
            return long_df

        d = long_df.copy()
        qc = mod_cfg.get("qc", {})
        mod_name = mod_cfg["name"]

        initial_count = len(d)

        # Fundus-specific QC
        if mod_name == "fundus_cd":
            qmin = qc.get("quality_min", None)
            if qmin is not None and "quality_grade" in d.columns:
                d = d[pd.to_numeric(d["quality_grade"], errors="coerce") >= qmin]

            allowed = qc.get("allowed_devices", [])
            if allowed and "device" in d.columns:
                d = d[d["device"].isin(allowed)]

        # RNFL-specific QC
        if mod_name == "rnfl":
            rng = qc.get("plausible_range_um", None)
            if rng:
                lo, hi = rng
                d.loc[(d["value"] < lo) | (d["value"] > hi), "value"] = np.nan

        # General QC: remove rows with invalid fraction threshold
        min_valid_frac = qc.get("min_valid_fraction_per_patient_eye", None)
        if min_valid_frac is not None:
            # Calculate valid fraction per patient-eye
            valid_mask = d["value"].notna()
            counts = d.groupby([self.k_patient, self.k_eye]).agg(
                total=("value", "count"),
                valid=("value", lambda x: x.notna().sum()),
            )
            counts["frac"] = counts["valid"] / counts["total"]
            valid_keys = counts[counts["frac"] >= min_valid_frac].index
            d = d[
                d.set_index([self.k_patient, self.k_eye]).index.isin(valid_keys)
            ].reset_index(drop=True)

        final_count = len(d)
        if initial_count > final_count:
            logger.info(
                f"QC for {mod_name}: {initial_count} -> {final_count} records "
                f"({initial_count - final_count} removed)"
            )

        return d

    def aggregate(self, long_df: pd.DataFrame, mod_cfg: dict) -> pd.DataFrame:
        """
        Aggregate data at specified level (e.g., patient-eye-visit).

        Args:
            long_df: Long-format data
            mod_cfg: Modality configuration

        Returns:
            Aggregated dataframe
        """
        if long_df.empty:
            return long_df

        agg_cfg = mod_cfg.get("aggregation", None)
        if not agg_cfg:
            return long_df

        level = agg_cfg.get("level", "patient_eye_visit")
        method = agg_cfg.get("method", "mean")

        if level != "patient_eye_visit":
            return long_df

        # Define grouping columns
        id_cols = [self.k_patient, self.k_eye, self.k_visit, "modality", "feature_full"]
        extra_keep = [
            c
            for c in long_df.columns
            if c not in id_cols + ["feature", "value"] and c != "feature_full"
        ]
        group_cols = [c for c in id_cols + extra_keep if c in long_df.columns]

        # Aggregate
        if method == "mean":
            out = long_df.groupby(group_cols, as_index=False)["value"].mean()
        elif method == "median":
            out = long_df.groupby(group_cols, as_index=False)["value"].median()
        else:
            out = long_df.groupby(group_cols, as_index=False)["value"].first()

        logger.debug(
            f"Aggregated {mod_cfg['name']}: {len(long_df)} -> {len(out)} records"
        )
        return out

    def merge_with_clinical(
        self, feat_long: pd.DataFrame, clin: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge feature data with clinical data.

        Args:
            feat_long: Long-format feature data
            clin: Clinical data

        Returns:
            Merged dataframe
        """
        merge_keys = [self.k_patient, self.k_eye, self.k_visit]
        merge_keys = [k for k in merge_keys if k in feat_long.columns and k in clin.columns]

        if not merge_keys:
            logger.warning("No common keys for merging; using patient_id only")
            merge_keys = [self.k_patient]

        merged = feat_long.merge(
            clin,
            on=merge_keys,
            how="left",
            suffixes=("", "_clin"),
        )

        logger.info(f"Merged data: {len(merged)} records")
        return merged

    def analyze_features(self, merged: pd.DataFrame) -> pd.DataFrame:
        """
        Run statistical analyses on all features.

        Analyses include:
        - Descriptive statistics by group
        - T-tests
        - Linear mixed models
        - AUROC

        Args:
            merged: Merged feature + clinical data

        Returns:
            Results dataframe with statistics per feature
        """
        cfg = self.cfg
        group_col = cfg["clinical"]["group_col"]
        pos = cfg["clinical"]["outcomes"]["glaucoma_label_value"]
        ctrl = cfg["clinical"]["outcomes"]["control_label_value"]
        feat_rules = cfg.get("feature_categories", {}).get("rules", [])

        results = []

        # Group by modality and feature
        for (modality, feat), g in merged.groupby(["modality", "feature_full"]):
            logger.debug(f"Analyzing: {feat}")

            # Split by group
            c = g[g[group_col].astype(str) == str(ctrl)]["value"]
            p = g[g[group_col].astype(str) == str(pos)]["value"]

            # Descriptive statistics
            c_mean = safe_mean(c)
            c_sd = safe_std(c)
            p_mean = safe_mean(p)
            p_sd = safe_std(p)

            # T-test
            try:
                c_clean = c.dropna()
                p_clean = p.dropna()
                if len(c_clean) >= 2 and len(p_clean) >= 2:
                    pval = stats.ttest_ind(c_clean, p_clean, equal_var=False).pvalue
                else:
                    pval = np.nan
            except Exception:
                pval = np.nan

            # Get modality config
            mod_cfg = next(
                (m for m in cfg["modalities"] if m["name"] == modality), None
            )

            # LMM analysis
            coef = np.nan
            ci_low = np.nan
            ci_high = np.nan
            lmm_p = np.nan

            if mod_cfg:
                lmm_cfg = mod_cfg.get("analysis", {}).get("lmm", {})
                if lmm_cfg.get("enabled", False):
                    dd = g.copy()

                    # Prepare categorical variables
                    for col in ["sex", "device", "eye"]:
                        if col in dd.columns:
                            dd[col] = dd[col].astype("category")

                    # Get formula and run LMM
                    formula = lmm_cfg.get("formula", "value ~ group")
                    group_re = lmm_cfg.get("random_effects", [self.k_patient])[0]

                    if group_re in dd.columns:
                        model = run_lmm(dd, formula=formula, group_col=group_re)
                        if model is not None:
                            # Extract group coefficient
                            term = f"{group_col}[T.{pos}]"
                            if term in model.params.index:
                                coef = float(model.params[term])
                                se = float(model.bse[term])
                                lmm_p = float(model.pvalues[term])
                                ci_low = coef - 1.96 * se
                                ci_high = coef + 1.96 * se

            # AUROC
            auroc = np.nan
            if mod_cfg:
                au_cfg = mod_cfg.get("analysis", {}).get("auroc", {})
                if au_cfg.get("enabled", False):
                    auroc = run_auroc(g, label_col=group_col, pos_label=pos)

            # Categorize feature
            category = categorize_feature(feat, feat_rules)

            results.append(
                {
                    "modality": modality,
                    "category": category,
                    "feature_full": feat,
                    "n_control": int(c.notna().sum()),
                    "n_glaucoma": int(p.notna().sum()),
                    "mean_control": c_mean,
                    "sd_control": c_sd,
                    "mean_glaucoma": p_mean,
                    "sd_glaucoma": p_sd,
                    "t_pvalue": pval,
                    "lmm_coef": coef,
                    "lmm_ci_low": ci_low,
                    "lmm_ci_high": ci_high,
                    "lmm_pvalue": lmm_p,
                    "auroc": auroc,
                }
            )

        res = pd.DataFrame(results)

        if res.empty:
            logger.warning("No features to analyze")
            return res

        # FDR correction within modality
        res["p_for_fdr"] = res["lmm_pvalue"].where(
            ~res["lmm_pvalue"].isna(), res["t_pvalue"]
        )
        res["p_fdr"] = np.nan

        for modality in res["modality"].unique():
            mask = res["modality"] == modality
            pvals = res.loc[mask, "p_for_fdr"].values
            valid = ~pd.isna(pvals)
            if valid.sum() > 0:
                _, qvals, _, _ = multipletests(pvals[valid], method="fdr_bh")
                res.loc[res[mask].index[valid], "p_fdr"] = qvals

        logger.info(f"Analyzed {len(res)} features")
        return res

    def make_table1(self, clin: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Table 1 (demographics and baseline characteristics).

        Args:
            clin: Clinical data

        Returns:
            Table 1 as a dataframe
        """
        cfg = self.cfg
        group_col = cfg["clinical"]["group_col"]
        ctrl = cfg["clinical"]["outcomes"]["control_label_value"]
        pos = cfg["clinical"]["outcomes"]["glaucoma_label_value"]

        cont = cfg["tables"]["table1_demographics"].get("continuous", [])
        cat = cfg["tables"]["table1_demographics"].get("categorical", [])

        rows = []

        # Sample size
        n_ctrl = (clin[group_col].astype(str) == str(ctrl)).sum()
        n_glc = (clin[group_col].astype(str) == str(pos)).sum()
        rows.append(
            {
                "Characteristic": "N",
                f"{ctrl}": str(n_ctrl),
                f"{pos}": str(n_glc),
                "P-value": "",
            }
        )

        # Continuous variables
        for col in cont:
            if col not in clin.columns:
                continue

            c = pd.to_numeric(
                clin[clin[group_col].astype(str) == str(ctrl)][col], errors="coerce"
            )
            g = pd.to_numeric(
                clin[clin[group_col].astype(str) == str(pos)][col], errors="coerce"
            )

            try:
                c_clean = c.dropna()
                g_clean = g.dropna()
                if len(c_clean) >= 2 and len(g_clean) >= 2:
                    pval = stats.ttest_ind(c_clean, g_clean, equal_var=False).pvalue
                else:
                    pval = np.nan
            except Exception:
                pval = np.nan

            rows.append(
                {
                    "Characteristic": col,
                    f"{ctrl}": f"{safe_mean(c):.2f} +/- {safe_std(c):.2f}",
                    f"{pos}": f"{safe_mean(g):.2f} +/- {safe_std(g):.2f}",
                    "P-value": format_p(pval),
                }
            )

        # Categorical variables
        for col in cat:
            if col not in clin.columns:
                continue

            try:
                tab = pd.crosstab(
                    clin[group_col].astype(str), clin[col].astype(str)
                )
                if tab.shape[0] >= 2 and tab.shape[1] >= 2:
                    pval = stats.chi2_contingency(tab.values)[1]
                else:
                    pval = np.nan
            except Exception:
                pval = np.nan
                tab = pd.DataFrame()

            ctrl_str = ""
            pos_str = ""
            if not tab.empty:
                if str(ctrl) in tab.index:
                    ctrl_str = "; ".join(
                        [f"{k}={v}" for k, v in tab.loc[str(ctrl)].to_dict().items()]
                    )
                if str(pos) in tab.index:
                    pos_str = "; ".join(
                        [f"{k}={v}" for k, v in tab.loc[str(pos)].to_dict().items()]
                    )

            rows.append(
                {
                    "Characteristic": col,
                    f"{ctrl}": ctrl_str,
                    f"{pos}": pos_str,
                    "P-value": format_p(pval),
                }
            )

        t1 = pd.DataFrame(rows)
        t1.to_csv(self.out_dir / "tables" / "Table1_Demographics.csv", index=False)
        logger.info("Generated Table 1 (Demographics)")

        return t1

    def export_master_tables(self, res: pd.DataFrame) -> None:
        """
        Export master tables with formatted statistics.

        Args:
            res: Results dataframe from analyze_features
        """
        if res.empty:
            logger.warning("No results to export")
            return

        out = res.copy()

        # Format columns
        out["Mean Control (+/-SD)"] = out.apply(
            lambda r: f"{r['mean_control']:.3f} +/- {r['sd_control']:.3f}"
            if pd.notna(r["mean_control"])
            else "",
            axis=1,
        )
        out["Mean Glaucoma (+/-SD)"] = out.apply(
            lambda r: f"{r['mean_glaucoma']:.3f} +/- {r['sd_glaucoma']:.3f}"
            if pd.notna(r["mean_glaucoma"])
            else "",
            axis=1,
        )
        out["LMM Coef"] = out["lmm_coef"].map(
            lambda x: "" if pd.isna(x) else f"{x:.4f}"
        )
        out["95% CI"] = out.apply(
            lambda r: ""
            if pd.isna(r["lmm_ci_low"])
            else f"[{r['lmm_ci_low']:.4f}, {r['lmm_ci_high']:.4f}]",
            axis=1,
        )
        out["P-value"] = out["lmm_pvalue"].map(format_p)
        out["FDR P-value"] = out["p_fdr"].map(format_p)
        out["AUROC"] = out["auroc"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

        # Select and order columns
        keep = [
            "modality",
            "category",
            "feature_full",
            "n_control",
            "n_glaucoma",
            "Mean Control (+/-SD)",
            "Mean Glaucoma (+/-SD)",
            "LMM Coef",
            "95% CI",
            "P-value",
            "FDR P-value",
            "AUROC",
        ]
        keep = [c for c in keep if c in out.columns]
        out = out[keep].sort_values(["modality", "category", "FDR P-value"])

        # Save combined table
        out.to_csv(
            self.out_dir / "tables" / "MasterTable_AllModalities.csv", index=False
        )
        logger.info("Saved MasterTable_AllModalities.csv")

        # Save per-modality tables
        for mod in out["modality"].unique():
            mod_df = out[out["modality"] == mod]
            mod_df.to_csv(
                self.out_dir / "tables" / f"MasterTable_{mod}.csv", index=False
            )
            logger.info(f"Saved MasterTable_{mod}.csv")

    def run(self) -> dict:
        """
        Execute the complete pipeline.

        Returns:
            Dictionary with pipeline outputs
        """
        logger.info("=" * 60)
        logger.info("Starting Multimodal Glaucoma AI Pipeline")
        logger.info("=" * 60)

        outputs = {}

        # Load clinical data
        clin = self.load_clinical()

        # Generate Table 1
        if self.cfg["tables"]["table1_demographics"].get("enabled", True):
            outputs["table1"] = self.make_table1(clin)

        # Process each modality
        all_long = []
        for mod_cfg in self.cfg["modalities"]:
            try:
                long_df = self.modality_to_long(mod_cfg)
                long_df = self.apply_qc(long_df, mod_cfg)
                long_df = self.aggregate(long_df, mod_cfg)
                all_long.append(long_df)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {mod_cfg['name']}: {e}")
            except Exception as e:
                logger.error(f"Error processing {mod_cfg['name']}: {e}")

        if not all_long:
            logger.error("No modality data loaded. Check data file paths.")
            return outputs

        # Combine all modalities
        feat_long = pd.concat(all_long, ignore_index=True)
        logger.info(f"Combined feature data: {len(feat_long)} records")

        # Merge with clinical data
        merged = self.merge_with_clinical(feat_long, clin)

        # Save merged data
        export_formats = self.cfg.get("export", {}).get("formats", ["csv", "parquet"])
        if "parquet" in export_formats:
            merged.to_parquet(self.out_dir / "merged_long.parquet", index=False)
            logger.info("Saved merged_long.parquet")
        if "csv" in export_formats:
            merged.to_csv(self.out_dir / "merged_long.csv", index=False)
            logger.info("Saved merged_long.csv")

        outputs["merged"] = merged

        # Run feature analyses
        res = self.analyze_features(merged)
        res.to_csv(self.out_dir / "tables" / "FeatureStats_All.csv", index=False)
        outputs["feature_stats"] = res

        # Export master tables
        self.export_master_tables(res)

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Outputs saved to: {self.out_dir}")
        logger.info("=" * 60)

        return outputs


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Multimodal Glaucoma AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config config.yaml
  python run_pipeline.py --config config.yaml --base-dir /path/to/project

The pipeline processes OCTA, Fundus, and RNFL data, merges with clinical
parameters, and outputs statistical analyses including Table 1 and
master tables with LMM coefficients, AUROC, and FDR-corrected p-values.
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        default=None,
        help="Base directory for resolving relative paths (default: config file directory)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())

    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = config_path.parent

    # Run pipeline
    pipeline = MultiModalPipeline(cfg, base_dir=base_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
