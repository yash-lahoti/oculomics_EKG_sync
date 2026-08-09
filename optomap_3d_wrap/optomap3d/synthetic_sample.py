"""Procedurally generate a synthetic, wide-field-fundus-*style* test image.

This is NOT a real patient photo and is not sourced from Optos or any
clinical dataset -- real optomap exports are either proprietary marketing
assets or gated behind dataset-use agreements (e.g. the TOP and PRIME-FP20
UWF datasets), so this generator exists purely to give the projection
pipeline a representative 200-degree-FOV circular input to run end-to-end
on. Drop a real optomap PNG/JPG export in its place at any time -- the rest
of the pipeline doesn't care where the image came from.
"""
from __future__ import annotations

import numpy as np
import cv2


def _bezier_points(p0, p1, p2, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def generate_synthetic_optomap(size: int = 1600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy = size / 2, size / 2
    R = size * 0.47

    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= R

    # Base fundus color: warm orange-red, slightly darker toward the edge
    falloff = np.clip(1.0 - 0.35 * (dist / R) ** 2, 0, 1)
    base_color = np.array([40, 90, 200], dtype=np.float32)  # BGR
    img[:] = base_color
    img *= falloff[..., None]

    # Low-frequency mottling for texture
    noise = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 40)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    img += (noise[..., None] - 0.5) * 25

    # Optic disc: bright, slightly nasal (offset) circle
    disc_center = np.array([cx + R * 0.32, cy])
    disc_r = R * 0.09
    dd = np.sqrt(((xx - disc_center[0]) ** 2 + (yy - disc_center[1]) ** 2))
    disc_mask = dd <= disc_r
    disc_edge = np.clip(1 - (dd - disc_r) / (disc_r * 0.6), 0, 1)
    disc_color = np.array([170, 210, 245], dtype=np.float32)
    img = img * (1 - disc_edge[..., None]) + disc_color * disc_edge[..., None]

    # Macula: subtle darker oval, temporal to the disc
    mac_center = np.array([cx - R * 0.22, cy])
    md = np.sqrt(((xx - mac_center[0]) / 1.3) ** 2 + (yy - mac_center[1]) ** 2)
    mac_dark = np.clip(1 - md / (R * 0.14), 0, 1) ** 2
    img -= mac_dark[..., None] * np.array([15, 25, 35], dtype=np.float32)

    # Vessels: arcades radiating/curving from the optic disc
    canvas = img.copy()
    n_major = 8
    for k in range(n_major):
        angle0 = 2 * np.pi * k / n_major + rng.uniform(-0.15, 0.15)
        start = disc_center + disc_r * 0.9 * np.array([np.cos(angle0), np.sin(angle0)])
        mid_r = R * rng.uniform(0.45, 0.7)
        bend = angle0 + rng.uniform(-0.9, 0.9)
        mid = disc_center + mid_r * np.array([np.cos(bend), np.sin(bend)])
        end_r = R * rng.uniform(0.85, 0.98)
        end = disc_center + end_r * np.array([np.cos(angle0 + rng.uniform(-0.3, 0.3)), np.sin(angle0 + rng.uniform(-0.3, 0.3))])

        pts = _bezier_points(start, mid, end, n=80).astype(np.int32)
        vessel_color = (60, 40, 150) if k % 2 == 0 else (50, 30, 130)  # artery/vein tone
        thickness = max(2, int(size * 0.006))
        for i in range(len(pts) - 1):
            t = i / len(pts)
            th = max(1, int(thickness * (1 - 0.6 * t)))
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), vessel_color, th, cv2.LINE_AA)

        # A couple of branch vessels off the main arcade
        for _ in range(2):
            bi = rng.integers(15, len(pts) - 10)
            branch_dir = rng.uniform(0, 2 * np.pi)
            b_end = pts[bi] + R * 0.15 * np.array([np.cos(branch_dir), np.sin(branch_dir)])
            b_pts = _bezier_points(pts[bi].astype(float), (pts[bi] + b_end) / 2, b_end, n=30).astype(np.int32)
            for i in range(len(b_pts) - 1):
                cv2.line(canvas, tuple(b_pts[i]), tuple(b_pts[i + 1]), vessel_color, max(1, thickness // 3), cv2.LINE_AA)

    img = canvas
    img = np.clip(img, 0, 255)
    img[~mask] = 0
    return img.astype(np.uint8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sample_synthetic_optomap.png")
    parser.add_argument("--size", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image = generate_synthetic_optomap(size=args.size, seed=args.seed)
    cv2.imwrite(args.out, image)
    print(f"Wrote synthetic sample optomap to {args.out}")
