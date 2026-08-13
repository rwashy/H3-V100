"""Runtime guard for H3 dynamic VBAR prefetch options."""

import logging
import types

LOGGER = logging.getLogger("H3V100PrefetchGuard")
PATCH_MARKER = "_h3_v100_prefetch_guard"
ORIGINAL_FORWARD_ATTR = "_h3_v100_prefetch_guard_original_forward"
OPTION = "prefetch_dynamic_vbars"
ADAPTIVE_OPTIONS = {
    "v100_h3_qkv_chunking": True,
    "v100_h3_qkv_chunk_tokens": 1024,
    "v100_h3_qkv_chunk_threshold": 38_000,
    "v100_h3_qkv_cache_trim_threshold_mb": 2048,
}


def _unwrap_our_forward(value):
    current = value
    seen = set()
    while current is not None:
        function = getattr(current, "__func__", current)
        if not getattr(function, PATCH_MARKER, False):
            return current
        identity = id(function)
        if identity in seen:
            raise RuntimeError("H3 runtime guard detected a cyclic V100 wrapper chain.")
        seen.add(identity)
        current = getattr(function, ORIGINAL_FORWARD_ATTR, None)
    raise RuntimeError("H3 runtime guard could not recover its original forward.")


def _make_forward(original_forward, enabled, adaptive_memory, experimental_fp16):
    def guarded_forward(
        self, x, timestep, context, transformer_options={},
        minimax_payload=None, **kwargs,
    ):
        if not isinstance(transformer_options, dict):
            return original_forward(
                x, timestep, context, transformer_options,
                minimax_payload=minimax_payload, **kwargs,
            )
        missing = object()
        guarded_values = {OPTION: bool(enabled)}
        if adaptive_memory:
            guarded_values.update(ADAPTIVE_OPTIONS)
        guarded_values["v100_h3_experimental_fp16_linear"] = bool(experimental_fp16)
        previous_values = {
            key: transformer_options.get(key, missing) for key in guarded_values
        }
        for key, value in guarded_values.items():
            transformer_options[key] = value
        try:
            return original_forward(
                x, timestep, context,
                transformer_options=transformer_options,
                minimax_payload=minimax_payload, **kwargs,
            )
        finally:
            for key, previous in previous_values.items():
                if previous is missing:
                    transformer_options.pop(key, None)
                else:
                    transformer_options[key] = previous

    setattr(guarded_forward, PATCH_MARKER, True)
    setattr(guarded_forward, ORIGINAL_FORWARD_ATTR, _unwrap_our_forward(original_forward))
    return guarded_forward


class H3RuntimePrefetchGuard:
    def patch(self, model, enabled=False, adaptive_memory=False, experimental_fp16=False):
        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        if not hasattr(diffusion_model, "_forward"):
            raise RuntimeError("H3 runtime prefetch guard expected diffusion_model._forward.")
        key = "diffusion_model._forward"
        existing = patched.object_patches.get(key)
        if existing is not None:
            function = getattr(existing, "__func__", existing)
            if not getattr(function, PATCH_MARKER, False):
                raise RuntimeError(f"H3 runtime prefetch guard found another patch at {key}.")
            base_forward = _unwrap_our_forward(existing)
            refreshed = True
        else:
            base_forward = _unwrap_our_forward(diffusion_model._forward)
            refreshed = False
        patched.add_object_patch(
            key,
            types.MethodType(
                _make_forward(
                    base_forward, enabled, adaptive_memory, experimental_fp16
                ), diffusion_model
            ),
        )
        LOGGER.info(
            "H3 runtime guard active: prefetch_dynamic_vbars=%s, adaptive_memory=%s, "
            "fp16_linear=%s, refreshed=%s.",
            bool(enabled), bool(adaptive_memory), bool(experimental_fp16), refreshed,
        )
        return (patched,)
