import math

import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def compute_residual(pressure, divergence, is_fluid,
                      face_mask_u, face_mask_v, alpha, reciprocal_beta, residual_out):
    """
    Compute residual r = (left + right + down + up + alpha * divergence) - beta * pressure
    for fluid cells, matching the exact masked-neighbor logic used in _jacobi_kernel.

    Non-fluid cells get residual 0 (they aren't part of the solve).

    Args:
        pressure : ndarray (H, W)
        divergence : ndarray (H, W)
        is_fluid : ndarray (H, W), bool
        face_mask_u : ndarray (H, W+1), bool
        face_mask_v : ndarray (H+1, W), bool
        alpha : float
        reciprocal_beta : float
        residual_out : ndarray (H, W), preallocated output buffer

    Return:
        residual_out : ndarray (H, W)
    """
    beta = 1.0 / reciprocal_beta
    height, width = pressure.shape
    for j in prange(height):
        for i in range(width):
            if not is_fluid[j, i]:
                residual_out[j, i] = 0.0
                continue
            left  = pressure[j, max(i - 1, 0)] if face_mask_u[j, i]     else pressure[j, i]
            right = pressure[j, min(i + 1, width - 1)] if face_mask_u[j, i + 1] else pressure[j, i]
            down  = pressure[max(j - 1, 0), i] if face_mask_v[j, i]     else pressure[j, i]
            up    = pressure[min(j + 1, height - 1), i] if face_mask_v[j + 1, i] else pressure[j, i]
            residual_out[j, i] = (left + right + down + up +
                                  alpha * divergence[j, i]) - beta * pressure[j, i]

    return residual_out
