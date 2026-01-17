#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recover origin / forward / up (and a plausible target) from a LookAt view matrix.

Assumptions (default):
- V is a 4x4 VIEW matrix that transforms world -> camera.
- Column-vector convention (like OpenGL / GLM math), stored in numpy as a normal 2D array.
- Right-handed camera where camera forward is -Z in camera space (OpenGL-style LookAt).

If your convention differs (row-major / left-handed / forward=+Z), adjust flags below.
"""

import numpy as np

# ---------- user config ----------
RIGHT_HANDED = True          # OpenGL-style typically True
FORWARD_IS_NEG_Z = True      # OpenGL-style typically True; DirectX-style often False (+Z)
COLUMN_VECTOR = True         # If you use row-vector convention, set False
TARGET_DISTANCE = 1.0        # target = origin + forward * TARGET_DISTANCE
# --------------------------------


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def recover_from_view(V: np.ndarray):
    """
    Returns:
      origin (3,)
      forward (3,)  : world-space viewing direction
      up (3,)       : world-space up direction (orthonormalized)
      right (3,)
      target (3,)   : a plausible point along the forward ray
      C2W (4,4)     : camera->world matrix (inverse of view)
    """
    V = np.asarray(V, dtype=np.float64)
    if V.shape != (4, 4):
        raise ValueError("V must be 4x4.")

    # If you are using row-vector convention, a common conversion is transposing the matrix
    # so we can treat it as column-vector form internally.
    Vin = V.T if not COLUMN_VECTOR else V

    # Robust: invert to get camera-to-world
    C2W = np.linalg.inv(Vin)

    # Camera origin in world space: translation column
    origin = C2W[:3, 3].copy()

    # Camera basis in world space: columns 0,1,2 = right, up, (camera +Z)
    right = normalize(C2W[:3, 0].copy())
    up = normalize(C2W[:3, 1].copy())
    cam_z = normalize(C2W[:3, 2].copy())  # camera's +Z axis in world

    # Forward direction depends on convention
    if FORWARD_IS_NEG_Z:
        forward = -cam_z
    else:
        forward = cam_z

    # Optional: re-orthonormalize (handles mild numerical drift)
    # Enforce right-handedness if desired
    right = normalize(right)
    up = normalize(up - right * np.dot(up, right))
    if RIGHT_HANDED:
        # Ensure right = normalize(forward x up) or up = normalize(right x forward)
        # Here we rebuild right to be consistent with forward and up
        right = normalize(np.cross(forward, up))
        up = normalize(np.cross(right, forward))
    else:
        # Left-handed variant (less common with LookAt in GL)
        right = normalize(np.cross(up, forward))
        up = normalize(np.cross(forward, right))

    target = origin + forward * float(TARGET_DISTANCE)

    # If we transposed input for row-vector usage, transpose outputs back if you need C2W in original layout
    Cout = C2W.T if not COLUMN_VECTOR else C2W

    return origin, forward, up, right, target, Cout


def main():
    # Example: replace this with your own 4x4 view matrix
    V = np.array([
        [-0.137283, -0.0319925, -0.990015, 4.05402],
        [ 2.71355e-08, 0.999478, -0.0322983, 1.61647],
        [ 0.990532, -0.00443408, -0.137213, -2.30652],
        [ 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    origin, forward, up, right, target, C2W = recover_from_view(V)

    np.set_printoptions(precision=6, suppress=True)
    print("View V:\n", V)
    print("\nRecovered:")
    print("origin :", origin)
    print("forward:", forward)
    print("up     :", up)
    print("right  :", right)
    print("target :", target)
    print("\nC2W (inv(V)):\n", C2W)


if __name__ == "__main__":
    main()
