"""Quick-look 3D preview of the wrapped globe using matplotlib -- no Blender
or GPU renderer required. Good for sanity-checking the projection before
opening the OBJ in Blender for the full interactive 3D Wrap-style view.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .sphere_projection import spherical_to_cartesian


# Rendered as an open "bowl" (only the captured theta range), not a closed
# sphere: matplotlib's 3D surface has no real z-buffer, and a fully closed
# sphere self-occludes incorrectly with its painter's-algorithm depth sort.
# An open bowl has no back side to fight with, so it renders correctly from
# any angle.
VIEWS = {
    "fundus_en_face": (-89, -90),
    "oblique_wrap": (-35, -60),
    "profile": (-10, 0),
}


def render_preview(
    equirect_bgr: np.ndarray,
    out_dir: str,
    fov_deg: float = 200.0,
    radius: float = 12.0,
    lat_segments: int = 90,
    lon_segments: int = 180,
):
    os.makedirs(out_dir, exist_ok=True)

    theta_max = np.radians(fov_deg / 2.0)
    theta = np.linspace(0.0, theta_max, lat_segments + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, lon_segments + 1)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    X, Y, Z = spherical_to_cartesian(Theta, Phi, radius)

    rgb = cv2.cvtColor(equirect_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]

    theta_c = (theta[:-1] + theta[1:]) / 2
    phi_c = (phi[:-1] + phi[1:]) / 2
    Tc, Pc = np.meshgrid(theta_c, phi_c, indexing="ij")
    v_idx = np.clip((Tc / np.pi * h).astype(int), 0, h - 1)
    u_idx = np.clip((Pc / (2 * np.pi) * w).astype(int), 0, w - 1)
    face_colors = rgb[v_idx, u_idx]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=face_colors, shade=False, antialiased=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    written = []
    for name, (elev, azim) in VIEWS.items():
        ax.view_init(elev=elev, azim=azim)
        path = os.path.join(out_dir, f"preview_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
        written.append(path)

    plt.close(fig)
    return written
