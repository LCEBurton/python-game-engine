from collections.abc import Callable
from typing import Any

OperatorFn = Callable[..., Any]

OPERATOR_REGISTRY: dict[tuple[str, str], OperatorFn] = {}

def register_operator(operation: str, backend: str, fn: OperatorFn):
    key = (operation, backend)
    
    if key in OPERATOR_REGISTRY:
        raise ValueError(
                f"Operator already registered: "
                f"operation={operation!r}, backend={backend!r}"
        )

    OPERATOR_REGISTRY[key] = fn

def get_operator(operation: str, backend: str) -> OperatorFn:
    key = (operation, backend)

    try:
        return OPERATOR_REGISTRY[key]
    except KeyError:
        available = [
            f"{op}/{b}"
            for op, b in OPERATOR_REGISTRY
        ]

        raise ValueError(
            f"Unknown operator "
            f"operation={operation!r}, backend={backend!r}. "
            f"Available: {available}"
        )
