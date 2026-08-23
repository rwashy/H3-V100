"""Shared, workflow-scoped diagnostic settings and low-disturbance probes."""

import hashlib
import logging
import subprocess
import time

import torch

DIAGNOSTICS_KEY = "v100_diagnostics_enabled"
DIAGNOSTICS_INTERVAL_KEY = "v100_diagnostics_interval"
FLASH_BENCHMARK_OCCURRED_KEY = "v100_flash_benchmark_occurred"

LOGGER = logging.getLogger("V100Diagnostics")
_block_stats = {}
_deferred_sample_indices = {}


def reset_deferred_block_samples(transformer_options):
    """Start one forward's GPU-only block sampling buffer."""
    transformer_options["v100_deferred_block_samples"] = []


def enqueue_deferred_block_sample(transformer_options, block_index, result):
    """Queue a tiny device sample without synchronizing or copying to CPU."""
    if not result.is_cuda or not result.is_floating_point() or result.numel() == 0:
        return
    flat = result.detach().reshape(-1)
    # Integer arithmetic gives stable, distributed positions without allocating
    # a full-size mask.  The gather stays on the current CUDA stream.
    count = min(64, flat.numel())
    key = (flat.device.type, flat.device.index, flat.numel(), count)
    indices = _deferred_sample_indices.get(key)
    if indices is None:
        indices = torch.arange(count, device=flat.device, dtype=torch.int64)
        if count > 1:
            indices.mul_((flat.numel() - 1) // (count - 1))
            indices[-1] = flat.numel() - 1
        _deferred_sample_indices[key] = indices
    sample = flat.index_select(0, indices).float()
    transformer_options.setdefault("v100_deferred_block_samples", []).append(
        (int(block_index), sample)
    )


def harvest_deferred_block_samples(transformer_options):
    """Read samples only after the enclosing H3 forward has completed."""
    queued = transformer_options.pop("v100_deferred_block_samples", [])
    if not queued:
        return
    for block_index, sample in queued:
        host = sample.cpu().numpy()
        digest = hashlib.blake2b(host.tobytes(), digest_size=8).hexdigest()
        finite = bool(torch.from_numpy(host).isfinite().all())
        absmax = float(abs(host).max()) if host.size else 0.0
        LOGGER.info(
            "V100 diagnostics deferred H3 block sample: block=%d, "
            "hash=%s, absmax=%.7g, finite=%s, samples=%d.",
            block_index, digest, absmax, finite, host.size,
        )


def _cuda_memory_mib(device):
    mib = 1024 ** 2
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated": torch.cuda.memory_allocated(device) / mib,
        "reserved": torch.cuda.memory_reserved(device) / mib,
        "peak_allocated": torch.cuda.max_memory_allocated(device) / mib,
        "peak_reserved": torch.cuda.max_memory_reserved(device) / mib,
        "free": free / mib,
        "total": total / mib,
    }


def _begin_task_memory_diagnostic(state, device, sequence):
    if device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)
    memory = _cuda_memory_mib(device)
    state["task_memory_start"] = memory
    state["task_index"] = state.get("task_index", 0) + 1
    LOGGER.info(
        "V100 diagnostics task memory start: task=%d, sequence=%d, device=%s, "
        "allocated=%.1f MiB, reserved=%.1f MiB, free=%.1f/%.1f MiB; "
        "CUDA peak counters reset for this generation.",
        state["task_index"], sequence, device, memory["allocated"],
        memory["reserved"], memory["free"], memory["total"],
    )


def _report_task_memory(state, device, sequence):
    if device.type != "cuda" or "task_memory_start" not in state:
        return
    start = state["task_memory_start"]
    memory = _cuda_memory_mib(device)
    LOGGER.info(
        "V100 diagnostics task memory checkpoint: task=%d, sequence=%d, device=%s, "
        "start_allocated/reserved=%.1f/%.1f MiB, "
        "current_allocated/reserved=%.1f/%.1f MiB, "
        "task_peak_allocated/reserved=%.1f/%.1f MiB, free=%.1f/%.1f MiB.",
        state["task_index"], sequence, device, start["allocated"],
        start["reserved"], memory["allocated"], memory["reserved"],
        memory["peak_allocated"], memory["peak_reserved"], memory["free"],
        memory["total"],
    )


def _harvest_cuda_events(state):
    remaining = []
    for start, end, block_index in state["pending"]:
        if end.query():
            elapsed = start.elapsed_time(end)
            state["cuda_ms"] += elapsed
            state["cuda_samples"] += 1
            per_block = state["per_block"].setdefault(
                block_index,
                {"calls": 0, "cpu_ms": 0.0, "cuda_ms": 0.0, "cuda_samples": 0},
            )
            per_block["cuda_ms"] += elapsed
            per_block["cuda_samples"] += 1
        else:
            remaining.append((start, end, block_index))
    state["pending"] = remaining


def _nvidia_smi_snapshot(device_index):
    fields = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,"
        "pcie.link.gen.current,pcie.link.width.current,"
        "clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,"
        "clocks_throttle_reasons.hw_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown,"
        "clocks_throttle_reasons.sw_thermal_slowdown"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--id={0}".format(device_index),
                "--query-gpu=" + fields, "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    names = fields.split(",")
    return dict(zip(names, values)) if len(values) == len(names) else None


def run_h3_block_diagnostic(block_index, x, interval, transformer_options, call):
    """Run one H3 block while collecting asynchronous compute/wait indicators."""
    device_index = x.device.index if x.is_cuda else -1
    key = (device_index, int(x.shape[0]))
    new_state = key not in _block_stats
    state = _block_stats.setdefault(
        key,
        {
            "calls": 0, "cpu_ms": 0.0, "gap_ms": 0.0, "gap_samples": 0,
            "cuda_ms": 0.0, "cuda_samples": 0, "last_exit": None,
            "pending": [], "per_block": {},
        },
    )
    entered = time.perf_counter()
    task_boundary = new_state
    if state["last_exit"] is not None:
        gap_ms = (entered - state["last_exit"]) * 1000.0
        if gap_ms > 10000.0:
            task_boundary = True
            state.update(
                {
                    "calls": 0, "cpu_ms": 0.0, "gap_ms": 0.0,
                    "gap_samples": 0, "cuda_ms": 0.0, "cuda_samples": 0,
                    "last_exit": None, "pending": [], "per_block": {},
                }
            )
        else:
            state["gap_ms"] += gap_ms
            state["gap_samples"] += 1

    if task_boundary:
        _begin_task_memory_diagnostic(state, x.device, key[1])

    if x.is_cuda:
        reclaimable = None
        try:
            import comfy_aimdo.model_vbar as model_vbar
            reclaimable = model_vbar.vbars_analyze(x.device.index) / (1024 ** 2)
        except (AttributeError, ImportError, RuntimeError, TypeError):
            pass
        allocated = torch.cuda.memory_allocated(x.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(x.device) / (1024 ** 2)
        free, _ = torch.cuda.mem_get_info(x.device)
        LOGGER.info(
            "V100 diagnostics H3 block entry: block=%d, allocated=%.1f MiB, "
            "reserved=%.1f MiB, driver_free=%.1f MiB, "
            "aimdo_reclaimable=%s.",
            block_index, allocated, reserved, free / (1024 ** 2),
            "unavailable" if reclaimable is None else f"{reclaimable:.1f} MiB",
        )
        total_mib = torch.cuda.get_device_properties(x.device).total_memory / (1024 ** 2)
        diagnostic_floor_mib = total_mib * 0.29
        if reclaimable is not None and free / (1024 ** 2) < diagnostic_floor_mib:
            try:
                import comfy.model_management as model_management
                candidates = []
                for loaded in model_management.current_loaded_models:
                    if loaded.device != x.device or loaded.is_dead():
                        continue
                    patcher = loaded.model
                    vbar_detail = ""
                    if patcher.is_dynamic():
                        vbar = patcher._vbar_get()
                        if vbar is not None:
                            residency = vbar.get_residency()
                            resident_pages = sum(bool(flags & 1) for flags in residency)
                            pinned_pages = sum(bool(flags & 2) for flags in residency)
                            vbar_detail = (
                                ",watermark=%.1f MiB,resident_pages=%d,pinned_pages=%d"
                                % (
                                    vbar.get_watermark() * 32.0,
                                    resident_pages, pinned_pages,
                                )
                            )
                    candidates.append(
                        "%s(dynamic=%s,loaded=%.1f MiB%s)" % (
                            type(patcher.model).__name__, patcher.is_dynamic(),
                            patcher.loaded_size() / (1024 ** 2), vbar_detail,
                        )
                    )
                LOGGER.info(
                    "V100 diagnostics AIMDO owners: block=%d, device=%s, models=%s.",
                    block_index, x.device,
                    ", ".join(candidates) if candidates else "none",
                )
            except (AttributeError, ImportError, RuntimeError):
                LOGGER.warning(
                    "V100 diagnostics AIMDO owner inspection unavailable: block=%d.",
                    block_index,
                )

    start = end = None
    if x.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    result = call()
    if end is not None:
        end.record()

    benchmark_occurred = bool(
        transformer_options.pop(FLASH_BENCHMARK_OCCURRED_KEY, False)
    )

    exited = time.perf_counter()
    cpu_ms = (exited - entered) * 1000.0
    state["calls"] += 1
    state["cpu_ms"] += cpu_ms
    state["last_exit"] = exited
    per_block = state["per_block"].setdefault(
        block_index,
        {"calls": 0, "cpu_ms": 0.0, "cuda_ms": 0.0, "cuda_samples": 0},
    )
    per_block["calls"] += 1
    per_block["cpu_ms"] += cpu_ms
    if start is not None and not benchmark_occurred:
        state["pending"].append((start, end, block_index))
        _harvest_cuda_events(state)

    if interval > 0 and state["calls"] % interval == 0:
        _report_task_memory(state, x.device, key[1])
        cuda_avg = state["cuda_ms"] / max(state["cuda_samples"], 1)
        cpu_avg = state["cpu_ms"] / state["calls"]
        gap_avg = state["gap_ms"] / max(state["gap_samples"], 1)
        slowest_cpu = max(
            state["per_block"].items(),
            key=lambda item: item[1]["cpu_ms"] / max(item[1]["calls"], 1),
        )
        completed_cuda = [
            (index, values["cuda_ms"] / values["cuda_samples"])
            for index, values in state["per_block"].items()
            if values["cuda_samples"] > 0
        ]
        slowest_cuda = max(completed_cuda, key=lambda item: item[1]) if completed_cuda else None
        LOGGER.info(
            "V100 diagnostics H3 block profile: calls=%d, sequence=%d, "
            "CPU-submit/wait avg=%.3f ms, inter-block gap avg=%.3f ms, "
            "CUDA avg=%.3f ms, CPU-minus-CUDA=%.3f ms "
            "(%d completed, %d pending), slowest CPU block=%d/%.3f ms, "
            "slowest CUDA block=%s.",
            state["calls"], key[1], cpu_avg, gap_avg, cuda_avg,
            cpu_avg - cuda_avg,
            state["cuda_samples"], len(state["pending"]), slowest_cpu[0],
            slowest_cpu[1]["cpu_ms"] / max(slowest_cpu[1]["calls"], 1),
            "n/a" if slowest_cuda is None else "%d/%.3f ms" % slowest_cuda,
        )
        telemetry = _nvidia_smi_snapshot(device_index) if device_index >= 0 else None
        if telemetry:
            LOGGER.info(
                "V100 diagnostics GPU telemetry: device=%d, gpu_util=%s%%, "
                "memory_util=%s%%, memory=%s/%s MiB, power=%s/%s W, temp=%s C, "
                "sm_clock=%s MHz, memory_clock=%s MHz, PCIe=Gen%s x%s, "
                "throttle_active=%s, sw_power_cap=%s, hw_slowdown=%s, "
                "hw_thermal=%s, sw_thermal=%s. "
                "PCIe link state is not real-time transfer throughput.",
                device_index, telemetry["utilization.gpu"],
                telemetry["utilization.memory"], telemetry["memory.used"],
                telemetry["memory.total"], telemetry["power.draw"],
                telemetry["power.limit"], telemetry["temperature.gpu"],
                telemetry["clocks.sm"], telemetry["clocks.mem"],
                telemetry["pcie.link.gen.current"],
                telemetry["pcie.link.width.current"],
                telemetry["clocks_throttle_reasons.active"],
                telemetry["clocks_throttle_reasons.sw_power_cap"],
                telemetry["clocks_throttle_reasons.hw_slowdown"],
                telemetry["clocks_throttle_reasons.hw_thermal_slowdown"],
                telemetry["clocks_throttle_reasons.sw_thermal_slowdown"],
            )
    return result
