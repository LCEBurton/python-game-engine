from numba import njit, prange

from cfd.backends.numba.operators.advection import sample_center_field

@njit(parallel=True, cache=True)
def restrict(fine, is_fluid_fine, coarse_out):
    """
    Restrict a fine-grid field (e.g. residual) to a coarse grid via
    mask-aware 2x2 block averaging.

    Args:
        fine : ndarray (H, W)
            Fine-grid field (e.g. residual), values in non-fluid cells ignored.
        is_fluid_fine : ndarray (H, W), bool
            Fluid mask on the fine grid.
        coarse_out : ndarray (H//2, W//2)
            Preallocated output buffer for coarse-grid field.

    Return:
        coarse_out : ndarray (H//2, W//2)
    """
    coarse_h, coarse_w = coarse_out.shape
    for j in prange(coarse_h):
        for i in range(coarse_w):
            j0, i0 = 2 * j, 2 * i
            total = 0.0
            count = 0
            for dj in range(2):
                for di in range(2):
                    jj, ii = j0 + dj, i0 + di
                    if is_fluid_fine[jj, ii]:
                        total += fine[jj, ii]
                        count += 1
            coarse_out[j, i] = total / count if count > 0 else 0.0

    return coarse_out

@njit(parallel=True, cache=True)
def prolong(coarse, is_fluid_fine, fine_out):
    """
    Prolong a coarse-grid field (e.g. error correction) to the fine grid
    via bilinear interpolation, reusing sample_center_field for cell-centered
    coordinate handling.

    Args:
        coarse : ndarray (H//2, W//2)
            Coarse-grid field (e.g. error).
        is_fluid_fine : ndarray (H, W), bool
            Fluid mask on the fine grid (correction skipped on non-fluid cells).
        fine_out : ndarray (H, W)
            Preallocated output buffer for fine-grid field.

    Return:
        fine_out : ndarray (H, W)
    """
    fine_h, fine_w = fine_out.shape

    for j in prange(fine_h):
        for i in range(fine_w):
            if not is_fluid_fine[j, i]:
                fine_out[j, i] = 0.0
                continue

            # Fine cell (i, j) center in fine physical space is (i+0.5, j+0.5).
            # Coarse grid has half the resolution, so coarse physical space
            # is fine physical space / 2.
            x_coarse = (i + 0.5) * 0.5
            y_coarse = (j + 0.5) * 0.5

            fine_out[j, i] = sample_center_field(coarse, x_coarse, y_coarse)

    return fine_out
