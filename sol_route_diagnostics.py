"""Bounded-memory diagnostics for the active Sol sparse-tail route."""

import torch


def summarize(selected, prefix_blocks):
    blocks = int(selected.shape[-1])
    tail = selected[..., int(prefix_blocks):, int(prefix_blocks):]
    width = int(tail.shape[-1])
    if not width or not tail.numel():
        return {"rows": 0, "blocks": width, "selected": 0,
                "density": 0.0, "density_min": 0.0, "density_max": 0.0,
                "nnz_mean": 0.0, "nnz_std": 0.0, "nnz_cv": 0.0,
                "nnz_quantiles": (0.0,) * 7, "all_dense_rows": 0,
                "run_count": 0, "avg_run_length": 0.0,
                "bitword_count": 0, "bitword_mean_popcount": 0.0,
                "bitword_empty_fraction": 0.0, "bitword_full_fraction": 0.0,
                "bitword_popcount_bands": (0,) * 6, "full_blocks": blocks}

    counts = tail.sum(-1, dtype=torch.int32)
    counts_f = counts.float()
    quantiles = tuple(float(v) for v in torch.quantile(
        counts_f, torch.tensor((0.0, .1, .25, .5, .75, .9, 1.0),
                               device=tail.device)).tolist())
    selected_count = int(counts.sum().item())
    rows = int(counts.numel())
    mean = float(counts_f.mean().item())
    std = float(counts_f.std(unbiased=False).item())

    runs = torch.zeros((), device=tail.device, dtype=torch.int64)
    for start in range(0, width, 256):
        part = tail[..., start:min(width, start + 256)]
        previous = tail[..., start - 1] if start else None
        runs.add_(part[..., 0].sum(dtype=torch.int64) if previous is None else
                  (part[..., 0] & ~previous).sum(dtype=torch.int64))
        if part.shape[-1] > 1:
            runs.add_((part[..., 1:] & ~part[..., :-1]).sum(dtype=torch.int64))
    run_count = int(runs.item())

    hist = torch.zeros(33, device=tail.device, dtype=torch.int64)
    words = 0
    for start in range(0, width, 32):
        popcount = tail[..., start:min(width, start + 32)].sum(-1, dtype=torch.int32)
        hist.add_(torch.bincount(popcount.reshape(-1), minlength=33))
        words += int(popcount.numel())
    values = tuple(int(v) for v in hist.tolist())
    bands = (values[0], sum(values[1:9]), sum(values[9:17]),
             sum(values[17:25]), sum(values[25:32]), values[32])
    return {
        "rows": rows, "blocks": width, "selected": selected_count,
        "density": selected_count / max(1, rows * width),
        "density_min": float(counts_f.min().item() / width),
        "density_max": float(counts_f.max().item() / width),
        "nnz_mean": mean, "nnz_std": std, "nnz_cv": std / max(mean, 1e-12),
        "nnz_quantiles": quantiles,
        "all_dense_rows": int((counts == width).sum().item()),
        "run_count": run_count,
        "avg_run_length": selected_count / max(1, run_count),
        "bitword_count": words,
        "bitword_mean_popcount": sum(i * n for i, n in enumerate(values)) / max(1, words),
        "bitword_empty_fraction": values[0] / max(1, words),
        "bitword_full_fraction": values[32] / max(1, words),
        "bitword_popcount_bands": bands, "full_blocks": blocks,
    }
