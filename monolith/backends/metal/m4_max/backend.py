from ..base import MetalBackend


class Backend(MetalBackend):
    """Native fallback pending measurements on M4 Max (32 or 40 cores)."""
    id = "m4_max"
