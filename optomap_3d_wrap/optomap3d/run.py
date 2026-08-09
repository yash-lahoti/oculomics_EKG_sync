"""CLI: project a 2D optomap-style image onto a 3D eye globe.

Usage
-----
    python -m optomap3d.run --input path/to/optomap.png --outdir out/

If --input is omitted, a synthetic sample image is generated first (see
optomap3d/synthetic_sample.py) so the pipeline can be demoed with no
external data.
"""
from __future__ import annotations

import argparse
import os

import cv2

from .sphere_projection import optomap_to_equirectangular, find_fundus_circle
from .mesh import build_uv_sphere, export_obj
from .preview_render import render_preview
from .synthetic_sample import generate_synthetic_optomap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Path to a 2D optomap image (PNG/JPG). Omit to use a generated synthetic sample.")
    parser.add_argument("--outdir", default="out", help="Directory to write outputs to.")
    parser.add_argument("--fov-deg", type=float, default=200.0, help="Total angular field of view of the source capture (Optos widefield ~200deg).")
    parser.add_argument("--eye-radius-mm", type=float, default=12.0, help="Approximate globe radius in mm (adult eye ~12mm axial radius) used for the exported mesh scale.")
    parser.add_argument("--equirect-width", type=int, default=2048)
    parser.add_argument("--equirect-height", type=int, default=1024)
    parser.add_argument("--lat-segments", type=int, default=96)
    parser.add_argument("--lon-segments", type=int, default=192)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.input is None:
        print("No --input given: generating a synthetic sample optomap (not real patient data).")
        image = generate_synthetic_optomap()
        args.input = os.path.join(args.outdir, "synthetic_input.png")
        cv2.imwrite(args.input, image)
    else:
        image = cv2.imread(args.input)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")

    cx, cy, r = find_fundus_circle(image)
    print(f"Detected fundus circle: center=({cx:.1f}, {cy:.1f}) radius={r:.1f}px")

    equirect = optomap_to_equirectangular(
        image,
        fov_deg=args.fov_deg,
        center=(cx, cy),
        radius=r,
        out_width=args.equirect_width,
        out_height=args.equirect_height,
    )
    texture_path = os.path.join(args.outdir, "equirect_texture.png")
    cv2.imwrite(texture_path, equirect)
    print(f"Wrote equirectangular texture: {texture_path}")

    vertices, uvs, faces = build_uv_sphere(
        radius=args.eye_radius_mm,
        lat_segments=args.lat_segments,
        lon_segments=args.lon_segments,
    )
    obj_path = os.path.join(args.outdir, "eye_globe.obj")
    export_obj(obj_path, vertices, uvs, faces, texture_filename=os.path.basename(texture_path))
    print(f"Wrote mesh: {obj_path} (+ .mtl). Import into Blender with File > Import > Wavefront (.obj).")

    previews = render_preview(equirect, args.outdir, fov_deg=args.fov_deg)
    print("Wrote preview renders:")
    for p in previews:
        print(f"  {p}")


if __name__ == "__main__":
    main()
