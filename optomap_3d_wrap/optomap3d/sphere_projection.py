"""Core math: reproject a 2D optomap (wide-field fundus) image onto a 3D eye globe.

An Optos-style optomap is (to a good approximation) an *azimuthal equidistant*
projection centered on the posterior pole: the distance of a pixel from the
image center is linearly proportional to the angle theta (measured from the
posterior pole / optical axis) that scan ray subtends at the eye, up to a
total field of view of ~200 degrees (i.e. theta_max = 100 degrees).

    r_pixels / R_pixels = theta / theta_max

That lets us invert the projection: for any direction (theta, phi) on the
eye globe we know exactly which source pixel to sample. We use that inverse
mapping to resample the flat image into an *equirectangular* image
(u = phi / 2pi, v = theta / pi), which is the texture layout a standard
lat/long UV sphere expects (and what Blender's "Sphere Projection" unwrap
produces), so the result can be applied straight onto a 3D globe.
"""
from __future__ import annotations

import numpy as np
import cv2


def find_fundus_circle(image: np.ndarray, thresh: int = 10) -> tuple[float, float, float]:
    """Auto-detect the circular imaged region (center + radius, in pixels).

    Optos exports are usually a circular fundus capture on a black
    background, sometimes with UI chrome around it. We threshold on
    brightness and take the largest connected blob's minimum enclosing
    circle. Falls back to "whole image inscribed circle" if nothing is
    found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(largest)
        if r > 10:
            return float(cx), float(cy), float(r)
    h, w = gray.shape[:2]
    return w / 2.0, h / 2.0, min(w, h) / 2.0


def optomap_to_equirectangular(
    image: np.ndarray,
    fov_deg: float = 200.0,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    out_width: int = 2048,
    out_height: int = 1024,
) -> np.ndarray:
    """Inverse azimuthal-equidistant -> equirectangular resample.

    Parameters
    ----------
    image: source optomap image (H, W, 3), BGR (as read by cv2.imread)
    fov_deg: total angular field of view captured by the optomap (Optos
        widefield/optomap devices are commonly quoted at ~200 degrees)
    center, radius: pixel center and pixel radius of the imaged circle in
        `image`. Auto-detected via `find_fundus_circle` if not given.
    out_width, out_height: size of the output equirectangular canvas. Height
        maps theta in [0, pi] (only the top fov_deg/2/180 fraction will be
        filled, the rest is the unseen far side of the globe -> left black).
        Width maps phi in [0, 2*pi).

    Returns
    -------
    Equirectangular image, shape (out_height, out_width, 3), same dtype as
    input, with unfilled (unseen) pixels left at 0.
    """
    if center is None or radius is None:
        cx, cy, r = find_fundus_circle(image)
        center = center or (cx, cy)
        radius = radius or r

    cx, cy = center
    theta_max = np.radians(fov_deg / 2.0)

    v_idx, u_idx = np.meshgrid(
        np.arange(out_height, dtype=np.float32),
        np.arange(out_width, dtype=np.float32),
        indexing="ij",
    )
    theta = (v_idx / out_height) * np.pi
    phi = (u_idx / out_width) * 2.0 * np.pi

    valid = theta <= theta_max
    r_src = (theta / theta_max) * radius
    map_x = (cx + r_src * np.cos(phi)).astype(np.float32)
    map_y = (cy + r_src * np.sin(phi)).astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    equirect = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return equirect


def spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray, radius: float = 1.0):
    """theta = angle from posterior pole (0..pi), phi = azimuth (0..2pi)."""
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = -radius * np.cos(theta)
    return x, y, z
