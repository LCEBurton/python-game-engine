import numpy as np
from numba import njit, prange

from engine.smoke_simulator.kernels.numba.advection import sample_center_field


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

@njit(parallel=True, cache=True)
def coarsen_solid_mask(solid_fine, solid_coarse_out):
    """
    Coarsen a cell-centered solid mask by 2x2 block reduction using
    'any fine solid -> coarse solid' rule (conservative, avoids leaking
    fluid through thin walls).

    Args:
        solid_fine : ndarray (H, W), bool
            Fine-grid solid mask (1=solid).
        solid_coarse_out : ndarray (H//2, W//2), bool
            Preallocated output buffer for coarse solid mask.

    Return:
        solid_coarse_out : ndarray (H//2, W//2), bool
    """
    coarse_h, coarse_w = solid_coarse_out.shape
    for J in prange(coarse_h):
        for I in range(coarse_w):
            j0, i0 = 2 * J, 2 * I
            solid_coarse_out[J, I] = (
                solid_fine[j0, i0] or solid_fine[j0, i0 + 1] or
                solid_fine[j0 + 1, i0] or solid_fine[j0 + 1, i0 + 1]
            )
    return solid_coarse_out


def build_face_masks(solid_mask, face_mask_u_out, face_mask_v_out):
    """
    Build face_mask_u/face_mask_v from a solid mask, matching the fine-level
    convention: a face is 'fluid' (True) if neither neighboring cell is solid.
    Boundary faces (outer edges) default to False (solid/closed).

    Args:
        solid_mask : ndarray (H, W), bool
            Cell-centered solid mask (1=solid).
        face_mask_u_out : ndarray (H, W+1), bool
            Preallocated output buffer, should be zero-initialized.
        face_mask_v_out : ndarray (H+1, W), bool
            Preallocated output buffer, should be zero-initialized.

    Return:
        face_mask_u_out, face_mask_v_out
    """
    face_mask_u_out[:, 1:-1] = ~(solid_mask[:, :-1] | solid_mask[:, 1:])
    face_mask_v_out[1:-1, :] = ~(solid_mask[:-1, :] | solid_mask[1:, :])
    return face_mask_u_out, face_mask_v_out


@njit(parallel=True, cache=True)
def compute_is_fluid(face_mask_u, face_mask_v, is_fluid_out):
    """
    Compute is_fluid mask from face masks, matching:
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    Args:
        face_mask_u : ndarray (H, W+1), bool
        face_mask_v : ndarray (H+1, W), bool
        is_fluid_out : ndarray (H, W), bool
            Preallocated output buffer.

    Return:
        is_fluid_out : ndarray (H, W), bool
    """
    height, width = is_fluid_out.shape
    for j in prange(height):
        for i in range(width):
            is_fluid_out[j, i] = (
                face_mask_u[j, i] or face_mask_u[j, i + 1] or
                face_mask_v[j, i] or face_mask_v[j + 1, i]
            )
    return is_fluid_out


def coarsen_level(solid_fine):
    """
    Convenience function: given a fine solid_mask, produce all coarse-level
    masks needed for one multigrid level.

    Args:
        solid_fine : ndarray (H, W), bool

    Return:
        solid_coarse : ndarray (H//2, W//2), bool
        face_mask_u_coarse : ndarray (H//2, W//2+1), bool
        face_mask_v_coarse : ndarray (H//2+1, W//2), bool
        is_fluid_coarse : ndarray (H//2, W//2), bool
    """
    h, w = solid_fine.shape
    coarse_h, coarse_w = h // 2, w // 2

    solid_coarse = np.zeros((coarse_h, coarse_w), dtype=np.bool_)
    coarsen_solid_mask(solid_fine, solid_coarse)

    face_mask_u_coarse = np.zeros((coarse_h, coarse_w + 1), dtype=np.bool_)
    face_mask_v_coarse = np.zeros((coarse_h + 1, coarse_w), dtype=np.bool_)
    build_face_masks(solid_coarse, face_mask_u_coarse, face_mask_v_coarse)

    is_fluid_coarse = np.zeros((coarse_h, coarse_w), dtype=np.bool_)
    compute_is_fluid(face_mask_u_coarse, face_mask_v_coarse, is_fluid_coarse)

    return solid_coarse, face_mask_u_coarse, face_mask_v_coarse, is_fluid_coarse


