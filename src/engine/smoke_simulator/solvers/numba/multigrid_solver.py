import numpy as np
from numba import njit, prange

from engine.smoke_simulator.kernels.numba.residual import compute_residual
from .multigrid_helper import restrict, prolong, compute_is_fluid, coarsen_solid_mask, build_face_masks
from .common import calculateAlpha, calculateReciprocalBeta
from ..registry_core import get_solver
from tools.profiling.session import ProfileSession



class MultigridSolver:
    """
    Geometric multigrid V-cycle pressure solver.
    Implements the PressureSolver protocol.

    The smoother used at each level is selected via `smoother_type`
    (any solver registered in SOLVER_REGISTRY, e.g. "jacobi", "gauss_seidel",
    "rb_sor_gauss_seidel").

    Caches per-level masks and buffers, only rebuilding when grid shape
    changes or `rebuild_masks=True` is passed (e.g. after solid geometry changes).
    """

    def __init__(self):
        self._cached_shape = None
        self._levels = list()
        self._buffers = list()

    def _build_levels(self, face_mask_u, face_mask_v, num_levels):
        height, width = face_mask_u.shape[0], face_mask_u.shape[1] - 1
        levels = []

        is_fluid = np.zeros((height, width), dtype=np.bool_)
        compute_is_fluid(face_mask_u, face_mask_v, is_fluid)
        solid = ~is_fluid

        levels.append({
            "is_fluid": is_fluid,
            "face_mask_u": face_mask_u,
            "face_mask_v": face_mask_v,
            "solid": solid,
        })

        for _ in range(1, num_levels):
            h, w = solid.shape
            if h % 2 != 0 or w % 2 != 0:
                break

            coarse_h, coarse_w = h // 2, w // 2
            solid_coarse = np.zeros((coarse_h, coarse_w), dtype=np.bool_)
            coarsen_solid_mask(solid, solid_coarse)

            fmu_coarse = np.zeros((coarse_h, coarse_w + 1), dtype=np.bool_)
            fmv_coarse = np.zeros((coarse_h + 1, coarse_w), dtype=np.bool_)
            build_face_masks(solid_coarse, fmu_coarse, fmv_coarse)

            is_fluid_coarse = np.zeros((coarse_h, coarse_w), dtype=np.bool_)
            compute_is_fluid(fmu_coarse, fmv_coarse, is_fluid_coarse)

            levels.append({
                "is_fluid": is_fluid_coarse,
                "face_mask_u": fmu_coarse,
                "face_mask_v": fmv_coarse,
                "solid": solid_coarse,
            })
            solid = solid_coarse

        return levels

    def _build_buffers(self, levels, dtype):
        buffers = []
        for level in levels:
            h, w = level["is_fluid"].shape
            buffers.append({
                "pressure": np.zeros((h, w), dtype=dtype),
                "residual": np.zeros((h, w), dtype=dtype),
                "rhs": np.zeros((h, w), dtype=dtype),
                "correction": np.zeros((h, w), dtype=dtype),
            })
        return buffers

    def _ensure_cache(self, face_mask_u, face_mask_v, num_levels, dtype, rebuild):
        shape = face_mask_u.shape
        if rebuild or self._cached_shape != shape or self._levels is None or len(self._levels) != num_levels:
            self._levels = self._build_levels(face_mask_u, face_mask_v, num_levels)
            self._buffers = self._build_buffers(self._levels, dtype)
            self._cached_shape = shape

    def _v_cycle(self, level_idx, pressure, rhs, fluid_density, cell_size, dt,
                 smoother_fn, smoother_iterations, coarsest_iterations, profiler):
        level = self._levels[level_idx]
        buf = self._buffers[level_idx]
        is_fluid = level["is_fluid"]
        face_mask_u = level["face_mask_u"]
        face_mask_v = level["face_mask_v"]

        is_coarsest = (level_idx == len(self._levels) - 1)
        n_iter = coarsest_iterations if is_coarsest else smoother_iterations
        scope_name = f"level_{level_idx}"

        with profiler.timing.scope(f"{scope_name}/pre_smooth"):
            smoother_fn(pressure, rhs, face_mask_u, face_mask_v, fluid_density, cell_size, dt, n_iter)

        if is_coarsest:
            return pressure

        alpha = calculateAlpha(cell_size, fluid_density, dt)
        reciprocal_beta = calculateReciprocalBeta()

        with profiler.timing.scope(f"{scope_name}/residual"):
            compute_residual(pressure, rhs, is_fluid, face_mask_u, face_mask_v,
                              alpha, reciprocal_beta, buf["residual"])

        coarse_buf = self._buffers[level_idx + 1]

        with profiler.timing.scope(f"{scope_name}/restrict"):
            restrict(buf["residual"], is_fluid, coarse_buf["rhs"])

        alpha_coarse = calculateAlpha(cell_size * 2.0, fluid_density, dt)
        coarse_buf["rhs"] /= alpha_coarse

        coarse_buf["pressure"].fill(0.0)
        self._v_cycle(level_idx + 1, coarse_buf["pressure"], coarse_buf["rhs"],
                      fluid_density, cell_size * 2.0, dt,
                      smoother_fn, smoother_iterations, coarsest_iterations, profiler)

        with profiler.timing.scope(f"{scope_name}/prolong"):
            prolong(coarse_buf["pressure"], is_fluid, buf["correction"])
            pressure += buf["correction"]

        with profiler.timing.scope(f"{scope_name}/post_smooth"):
            smoother_fn(pressure, rhs, face_mask_u, face_mask_v, fluid_density, cell_size, dt, n_iter)

        return pressure

    def __call__(
        self,
        pressure: np.ndarray,
        divergence: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        fluid_density: float,
        cell_size: float,
        dt: float,
        v_cycles: int = 2,
        smoother_type: str = "jacobi",
        num_levels: int = 4,
        smoother_iterations: int = 2,
        coarsest_iterations: int = 50,
        rebuild_masks: bool = False,
        profiler=None,
        **solver_kwargs,
    ) -> None:
        if profiler is None:
            profiler = ProfileSession(enabled=False)

        self._ensure_cache(face_mask_u, face_mask_v, num_levels, pressure.dtype, rebuild_masks)
        smoother_fn = get_solver(smoother_type)

        fine_buf = self._buffers[0]
        fine_buf["pressure"][:] = pressure
        fine_buf["rhs"][:] = divergence

        for _ in range(v_cycles):
            with profiler.timing.scope("v_cycle"):
                self._v_cycle(0, fine_buf["pressure"], fine_buf["rhs"],
                              fluid_density, cell_size, dt,
                              smoother_fn, smoother_iterations, coarsest_iterations, profiler)

        pressure[:] = fine_buf["pressure"]

