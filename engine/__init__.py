"""
engine package for Universal Test-Time Training (uTTT) checkpoint and personalization management.
"""

from importlib import import_module

__all__ = [
    "DualTrackCheckpointManager",
    "EvolutionResult",
    "format_ttt_status_bar",
    "uTTTPersonalizer",
]


def __getattr__(name):
    # Prompt formatting and the WebSocket-only client require no Torch install.
    # Preserve existing public imports, loading model dependencies on demand.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".checkpoint_manager" if name == "DualTrackCheckpointManager" else ".personalizer"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
