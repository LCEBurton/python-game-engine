from .base import PressureSolver

SOLVER_REGISTRY: dict[tuple[str, str], PressureSolver] = {}


def register_solver(method: str, backend: str, fn):
    key = (method, backend)
    
    if key in SOLVER_REGISTRY:
        raise ValueError(
                f"Pressure solver already registered: "
                f"method={method!r}, backend={backend!r}"
        )

    SOLVER_REGISTRY[key] = fn


def get_solver(method: str, backend: str):
    key = (method, backend)

    try:
        return SOLVER_REGISTRY[key]
    except KeyError:
        available = [
            f"{m}/{b}"
            for m, b in SOLVER_REGISTRY
        ]

        raise ValueError(
            f"Unknown pressure solver "
            f"method={method!r}, backend={backend!r}. "
            f"Available: {available}"
        )
