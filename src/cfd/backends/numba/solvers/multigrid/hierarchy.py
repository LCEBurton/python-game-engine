import numpy as np

from .masks import coarsen_solid_mask, build_face_masks, compute_is_fluid

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


