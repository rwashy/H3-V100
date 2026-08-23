"""Runtime guard for H3 dynamic VBAR prefetch options."""

import logging
import types

from .diagnostics import (
    DIAGNOSTICS_KEY,
    harvest_deferred_block_samples,
    reset_deferred_block_samples,
)
from .sol_attention import H3_VIDEO_GRID_KEY, PREFIX_STOP_KEY


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
_reported = set()
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


def _make_forward(
    original_forward, enabled, adaptive_memory, experimental_fp16,
):
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
        layout = (minimax_payload or {}).get("layout")
        prefix_stop = 0
        video_grid = None
        if layout is not None:
            prefix_stop = next(
                (
                    int(start) for start, _stop, kind in layout.segments
                    if kind == "video"
                ),
                0,
            )
            signature = getattr(layout, "signature", None)
            if isinstance(signature, (tuple, list)) and len(signature) >= 4:
                latent_t, latent_h, latent_w = signature[1:4]
                if all(
                    isinstance(value, int) and value > 0
                    for value in (latent_t, latent_h, latent_w)
                ):
                    video_grid = (
                        int(latent_t), int(latent_h) // 2, int(latent_w) // 2
                    )
        guarded_values[PREFIX_STOP_KEY] = prefix_stop
        guarded_values[H3_VIDEO_GRID_KEY] = video_grid
        previous_values = {
            key: transformer_options.get(key, missing) for key in guarded_values
        }
        for key, value in guarded_values.items():
            transformer_options[key] = value
        model_management = None
        previous_num_streams = None
        if adaptive_memory:
            try:
                import comfy.model_management as model_management
                previous_num_streams = model_management.NUM_STREAMS
                model_management.NUM_STREAMS = 0
            except (AttributeError, ImportError):
                model_management = None
        try:
            if transformer_options.get(DIAGNOSTICS_KEY, False):
                key = (
                    id(self), bool(enabled), bool(adaptive_memory),
                    bool(experimental_fp16),
                )
                if key not in _reported:
                    _reported.add(key)
                    LOGGER.info(
                        "V100 diagnostics H3 runtime prefetch guard: "
                        "prefetch_dynamic_vbars=%s, incoming=%s. The runtime "
                        "value is applied immediately before make_prefetch_queue.",
                        bool(enabled),
                        "unset" if previous_values[OPTION] is missing else previous_values[OPTION],
                    )
                    if adaptive_memory:
                        LOGGER.info(
                            "V100 diagnostics H3 runtime adaptive guard: "
                            "qkv_and_outproj_chunking=True, threshold=%d, chunk_tokens=%d.",
                            ADAPTIVE_OPTIONS["v100_h3_qkv_chunk_threshold"],
                            ADAPTIVE_OPTIONS["v100_h3_qkv_chunk_tokens"],
                        )
                    if experimental_fp16:
                        LOGGER.warning(
                            "V100 diagnostics H3 FP16 runtime guard: "
                            "mlp_fc2_scale=256; attention_out_proj=FP32."
                        )
                    LOGGER.info(
                        "V100 diagnostics H3 Sol-Attn layout guard: "
                        "condition_prefix_stop=%d, video_grid=%s, source=%s.",
                        prefix_stop, video_grid,
                        "minimax_payload.layout" if layout is not None
                        else "unavailable",
                    )
            if transformer_options.get(DIAGNOSTICS_KEY, False):
                reset_deferred_block_samples(transformer_options)
            result = original_forward(
                x, timestep, context,
                transformer_options=transformer_options,
                minimax_payload=minimax_payload, **kwargs,
            )
            if transformer_options.get(DIAGNOSTICS_KEY, False):
                streams = result if isinstance(result, (tuple, list)) else (result,)
                values = []
                for stream in streams:
                    if not hasattr(stream, "is_floating_point") or not stream.is_floating_point():
                        continue
                    finite = bool(stream.isfinite().all())
                    absmax = float(stream.float().abs().max()) if stream.numel() else 0.0
                    values.append((tuple(stream.shape), str(stream.dtype), finite, absmax))
                LOGGER.info("V100 diagnostics H3 model outputs: streams=%s.", values)
                # The model-output finite/absmax probe above is already the
                # forward-end synchronization boundary. Harvesting here adds
                # no block-level or pre-execution synchronization.
                harvest_deferred_block_samples(transformer_options)
            return result
        finally:
            if model_management is not None and previous_num_streams is not None:
                model_management.NUM_STREAMS = previous_num_streams
            for key, previous in previous_values.items():
                if previous is missing:
                    transformer_options.pop(key, None)
                else:
                    transformer_options[key] = previous

    setattr(guarded_forward, PATCH_MARKER, True)
    setattr(guarded_forward, ORIGINAL_FORWARD_ATTR, _unwrap_our_forward(original_forward))
    return guarded_forward


class H3RuntimePrefetchGuard:
    def patch(
        self, model, enabled=False, adaptive_memory=False,
        experimental_fp16=False,
    ):
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
                    base_forward, enabled, adaptive_memory, experimental_fp16,
                ), diffusion_model
            ),
        )
        LOGGER.info(
            "H3 runtime guard active: prefetch_dynamic_vbars=%s, adaptive_memory=%s, "
            "synchronous_cast_streams=%s, experimental_fp16=%s, refreshed=%s.",
            bool(enabled), bool(adaptive_memory), bool(adaptive_memory),
            bool(experimental_fp16), refreshed,
        )
        return (patched,)
