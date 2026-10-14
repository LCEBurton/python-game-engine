from numba import njit, prange

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

