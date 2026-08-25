"""No-op workflow marker for ComfyUI's native DynamicVRAM headroom policy."""


CONTROLLER_KEY = "v100_h3_dynamic_vbar_controller"


class NativeDynamicVBARPolicy:
    """Keep wrapper identity/selection guards without manually evicting pages."""

    def __init__(self, requested_headroom_gib):
        self.requested_headroom_gib = float(requested_headroom_gib)
        self.invocations = 0

    def begin_mlp(self, block_index, device=None):
        self.invocations += 1

    def reserve(self, projection, device):
        return 0
