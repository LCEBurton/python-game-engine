SOLVER_REGISTRY = {}


def register_solver(name: str, fn):
    SOLVER_REGISTRY[name] = fn


def get_solver(name: str):
    try:
        return SOLVER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown pressure solver: {name!r}. Available: {list(SOLVER_REGISTRY)}")
