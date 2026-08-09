# optomap3d — open-source optomap → 3D eye globe projection

Wraps a 2D ultra-widefield ("optomap"-style) retinal image onto a 3D eye
globe mesh, mimicking the effect of Optos's proprietary 3D Wrap, using only
open-source Python (NumPy/OpenCV) + a standard OBJ mesh you can open in
Blender (or any 3D viewer).

## Why this approach

Optos widefield devices capture up to ~200° of the retina in a single
scan. That flat capture is, to a very good approximation, an **azimuthal
equidistant projection** centered on the posterior pole: a pixel's distance
from the image center is linearly proportional to the angle it subtends at
the eye (0° at the center, ~100° at the edge for a 200° total FOV).

That means the projection is invertible in closed form — no ML, no
proprietary calibration needed:

```
theta = (r_pixels / R_pixels) * theta_max      # angle from posterior pole
phi   = atan2(dy, dx)                          # azimuth around the optical axis
x = R_eye * sin(theta) * cos(phi)
y = R_eye * sin(theta) * sin(phi)
z = -R_eye * cos(theta)
```

`optomap3d/sphere_projection.py` uses this to resample the flat optomap
into an **equirectangular** image (`u = phi / 2pi`, `v = theta / pi`) via
`cv2.remap`. Equirectangular is exactly the UV layout a standard lat/long
UV sphere uses, so `optomap3d/mesh.py` builds that sphere with matching UVs
and writes an OBJ+MTL with the texture already wired up — no manual
"Sphere Projection" unwrap step in Blender required, you can just import
and go.

## Install

```bash
cd optomap_3d_wrap
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# With a real optomap export:
python -m optomap3d.run --input /path/to/optomap.png --outdir out/

# Or, with no input, to try it on a generated demo image:
python -m optomap3d.run --outdir out/
```

Outputs in `out/`:

| File | What it is |
| --- | --- |
| `equirect_texture.png` | The optomap resampled into equirectangular layout |
| `eye_globe.obj` / `.mtl` | UV sphere mesh with the texture pre-wired — **File > Import > Wavefront (.obj)** in Blender |
| `preview_*.png` | Quick matplotlib 3D renders (posterior pole, 3/4, side views) so you can sanity-check the wrap without opening Blender |

Useful flags: `--fov-deg` (defaults to 200, matching Optos widefield/optomap
capture), `--eye-radius-mm` (defaults to 12mm, adult axial globe radius),
`--equirect-width/--equirect-height`, `--lat-segments/--lon-segments` (mesh
resolution).

## About the sample image

No real Optos/optomap image is bundled here. Real patient exports are
either proprietary marketing assets or gated behind dataset-use agreements
(e.g. the Tsukazaki Optos Public dataset, PRIME-FP20 on IEEE DataPort), so
they're not something to redistribute in an open-source repo. Instead,
`optomap3d/synthetic_sample.py` procedurally generates a **synthetic**
200°-FOV fundus-style test image (optic disc, macula, radiating vessel
arcades, circular vignette) purely from code, with no external data —
enough to exercise the whole pipeline end to end. Swap in a real optomap
PNG/JPG via `--input` at any time; the math doesn't change.

## Limitations / notes

- The azimuthal-equidistant model is Optos's documented approximate
  projection model for optomap; real captures have some device-specific
  nonlinearity/distortion near the far periphery that this doesn't
  attempt to reverse-engineer (Optos's exact "ProView" calibration isn't
  public).
- `find_fundus_circle` auto-detects the circular capture region by
  thresholding; pass `--input` on a tightly-cropped, UI-chrome-free export
  for the cleanest result, or hardcode center/radius in
  `sphere_projection.optomap_to_equirectangular` if auto-detection picks
  the wrong blob (e.g. due to on-image annotations).
- This produces a single hemisphere-ish wrap from one capture. Optos's own
  3D Wrap output is typically built from multiple montaged captures (or a
  single 200° capture) mapped the same way — feed in whichever flat image
  you have.
