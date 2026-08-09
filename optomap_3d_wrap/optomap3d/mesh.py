"""Build a lat/long UV sphere and export it (with the equirectangular optomap
texture already applied) as an OBJ+MTL that Blender / MeshLab / any viewer
can open directly -- no manual "Sphere Projection" unwrap step needed,
since the UVs are written out already matching the texture.
"""
from __future__ import annotations

import os
import numpy as np


def build_uv_sphere(radius: float = 12.0, lat_segments: int = 96, lon_segments: int = 192):
    """Returns (vertices, uvs, faces). v = theta/pi (0 = posterior pole),
    u = phi/2pi, matching optomap_to_equirectangular's layout.
    """
    thetas = np.linspace(0.0, np.pi, lat_segments + 1)
    phis = np.linspace(0.0, 2.0 * np.pi, lon_segments + 1)

    vertices = []
    uvs = []
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = -radius * np.cos(theta)
            vertices.append((x, y, z))
            uvs.append((j / lon_segments, 1.0 - i / lat_segments))

    faces = []
    row_stride = lon_segments + 1
    for i in range(lat_segments):
        for j in range(lon_segments):
            v00 = i * row_stride + j
            v01 = i * row_stride + (j + 1)
            v10 = (i + 1) * row_stride + j
            v11 = (i + 1) * row_stride + (j + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    return np.array(vertices), np.array(uvs), faces


def export_obj(out_path: str, vertices, uvs, faces, texture_filename: str):
    base = os.path.splitext(out_path)[0]
    mtl_path = base + ".mtl"
    mtl_name = os.path.basename(mtl_path)
    material_name = "optomap_material"

    with open(mtl_path, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {texture_filename}\n")

    with open(out_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o eye_globe_retina\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write(f"usemtl {material_name}\n")
        for a, b, c in faces:
            # OBJ indices are 1-based; texture index == vertex index here.
            f.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")
