"""Release inactive DynamicVRAM models at the H3 sampling phase boundary."""

import logging
import threading


LOGGER = logging.getLogger("H3V100PhaseRelease")
MODEL_MARKER = "_h3_v100_phase_boundary_managed"
LOAD_GUARD_MARKER = "_h3_v100_load_phase_guard"
LOAD_GUARD_ORIGINAL = "_h3_v100_load_phase_guard_original"
_install_lock = threading.Lock()


def _is_cuda_dynamic(model):
    is_dynamic = getattr(model, "is_dynamic", None)
    if not callable(is_dynamic) or not bool(is_dynamic()):
        return False
    device = getattr(model, "load_device", None)
    return getattr(device, "type", None) == "cuda"


class PriorStageDynamicModelReleaser:
    """Detach inactive GPU Dynamic ModelPatchers at each H3 phase entry.

    ComfyUI marks models from an earlier execution phase as not currently used
    when it prepares the sampler's active model set. Dynamic-to-dynamic loads
    intentionally leave those models resident for demand paging. On a 16 GiB
    V100 that can leave too little physical space for the first H3 weight page.
    This selector is role/state based: no model class or model name is used.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._calls = 0

    def begin_h3_forward(self):
        with self._lock:
            self._calls += 1

            import comfy.model_management as model_management

            all_models = list(model_management.loaded_models())
            active_models = list(
                model_management.loaded_models(only_currently_used=True)
            )
            active_ids = {id(model) for model in active_models}
            candidates = [
                model for model in all_models
                if id(model) not in active_ids and _is_cuda_dynamic(model)
            ]

            released = []
            for model in candidates:
                model_type = type(getattr(model, "model", None)).__name__
                loaded_before = int(model.loaded_size())
                model_management.unload_model_and_clones(
                    model,
                    unload_additional_models=False,
                    all_devices=True,
                )
                remaining_ids = {
                    id(item) for item in model_management.loaded_models()
                }
                if id(model) in remaining_ids:
                    raise RuntimeError(
                        "H3 prior-stage dynamic model release did not detach "
                        f"an inactive CUDA ModelPatcher ({model_type})."
                    )
                released.append((model_type, loaded_before))

            # A cached MODEL object survives across prompts. Re-scan on every
            # H3 forward so a text encoder loaded by a later prompt is still
            # released. Keep the normal no-op path quiet after the first scan.
            if released or self._calls == 1:
                LOGGER.info(
                    "H3 prior-stage dynamic model release: selector="
                    "inactive_dynamic_cuda, class_name_filter=False, scan=%d, "
                    "candidates=%d, released=%d, released_loaded=%.1f MiB, "
                    "models=%s.",
                    self._calls,
                    len(candidates),
                    len(released),
                    sum(item[1] for item in released) / (1024 ** 2),
                    [item[0] for item in released],
                )
            return {
                "scan": self._calls,
                "candidates": len(candidates),
                "released": len(released),
                "released_loaded_mib": (
                    sum(item[1] for item in released) / (1024 ** 2)
                ),
                "models": tuple(item[0] for item in released),
            }


def mark_phase_managed(model):
    """Mark an optimized H3 ModelPatcher for cross-prompt phase release."""
    setattr(model, MODEL_MARKER, True)
    return model


def _model_key(model):
    clone_uuid = getattr(model, "clone_base_uuid", None)
    return ("clone", clone_uuid) if clone_uuid is not None else ("id", id(model))


def _requested_model_keys(models):
    requested = set()
    for model in models:
        requested.add(_model_key(model))
        nested = getattr(model, "model_patches_models", None)
        if callable(nested):
            requested.update(_model_key(item) for item in nested())
    return requested


def install_load_phase_guard():
    """Release an inactive optimized H3 before the next model is loaded.

    ComfyUI intentionally keeps DynamicVRAM models resident when switching to
    another DynamicVRAM model. That is normally useful, but after an H3 pass it
    can leave too little physical VRAM for the next prompt's text-encoder VBAR
    page. AIMDO aborts the process instead of raising a recoverable Python OOM.
    The guard is installed once and only evicts ModelPatchers explicitly marked
    by this node; unrelated DynamicVRAM models keep ComfyUI's normal policy.
    """
    with _install_lock:
        import comfy.model_management as model_management

        current = model_management.load_models_gpu
        if getattr(current, LOAD_GUARD_MARKER, False):
            return False

        def guarded_load_models_gpu(models, *args, **kwargs):
            models = list(models)
            requested_keys = _requested_model_keys(models)
            stale = [
                model for model in list(model_management.loaded_models())
                if getattr(model, MODEL_MARKER, False)
                and _model_key(model) not in requested_keys
            ]
            if stale:
                try:
                    import comfy.model_prefetch as model_prefetch
                    model_prefetch.cleanup_prefetch_queues()
                except (AttributeError, ImportError):
                    pass

                released = []
                for model in stale:
                    loaded_before = int(model.loaded_size())
                    model_type = type(getattr(model, "model", None)).__name__
                    model_management.unload_model_and_clones(
                        model,
                        unload_additional_models=False,
                        all_devices=True,
                    )
                    released.append((model_type, loaded_before))
                LOGGER.info(
                    "H3 next-prompt phase release: released=%d, "
                    "released_loaded=%.1f MiB, models=%s.",
                    len(released),
                    sum(item[1] for item in released) / (1024 ** 2),
                    [item[0] for item in released],
                )
            return current(models, *args, **kwargs)

        setattr(guarded_load_models_gpu, LOAD_GUARD_MARKER, True)
        setattr(guarded_load_models_gpu, LOAD_GUARD_ORIGINAL, current)
        model_management.load_models_gpu = guarded_load_models_gpu
        return True
