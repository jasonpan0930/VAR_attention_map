from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_attention_heatmap(
    attn: torch.Tensor,
    out_path: Path,
    title: str = "",
    query_idx: int = -1,
    head: Optional[int] = None,
    vmax: Optional[float] = None,
) -> None:
    """Save one attention map.

    attn: [H, Lq, Lk] or [Lq, Lk]
    """
    if attn.ndim == 3:
        if head is None:
            attn_2d = attn.mean(0)
            head_label = "mean"
        else:
            attn_2d = attn[head]
            head_label = str(head)
    else:
        attn_2d = attn
        head_label = "single"

    if query_idx < 0:
        query_idx = attn_2d.shape[0] + query_idx
    vec = attn_2d[query_idx].numpy()

    _ensure_dir(out_path)
    plt.figure(figsize=(10, 2.8))
    plt.imshow(vec[None, :], aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    plt.colorbar(fraction=0.025, pad=0.02)
    plt.yticks([0], [f"q={query_idx}"])
    plt.xlabel("key token index")
    if title:
        plt.title(f"{title} | head={head_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_stage_grid(
    attn: torch.Tensor,
    out_path: Path,
    title: str,
    query_idx: int = -1,
    max_heads: int = 4,
) -> None:
    """Plot a few heads side-by-side for one query token."""
    H = attn.shape[0]
    heads = list(range(min(max_heads, H)))

    _ensure_dir(out_path)
    fig, axes = plt.subplots(1, len(heads), figsize=(4 * len(heads), 3), squeeze=False)
    q = attn.shape[1] + query_idx if query_idx < 0 else query_idx
    for ax, h in zip(axes[0], heads):
        vec = attn[h, q].numpy()
        im = ax.imshow(vec[None, :], aspect="auto", cmap="magma", vmin=0.0)
        ax.set_title(f"head {h}")
        ax.set_yticks([])
        ax.set_xlabel("key")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle(f"{title} | query={q}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def cumulative_scale_bounds(patch_nums: Sequence[int], stage_idx: int) -> list[int]:
    """Return key-axis boundaries between cached scales (exclusive upper bounds)."""
    bounds = []
    cur = 0
    for si, pn in enumerate(patch_nums):
        cur += pn * pn
        if si < stage_idx:
            bounds.append(cur)
    return bounds


def save_qkt_mag_heatmap(
    qkt_mag: torch.Tensor,
    out_path: Path,
    stage_idx: int,
    patch_num: int,
    block_idx: int,
    patch_nums: Sequence[int],
    call_idx: int = 0,
    head: Optional[int] = None,
) -> None:
    """Plot QK^T magnitude heatmap in the VAR paper / debug style."""
    if qkt_mag.ndim == 3:
        mat = qkt_mag.mean(0).numpy() if head is None else qkt_mag[head].numpy()
    else:
        mat = qkt_mag.numpy()

    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")

    if head is None:
        title = f"QKT mag | stage{stage_idx:02d}_pn{patch_num}_block{block_idx:02d} | call {call_idx}"
        cbar_label = "mean QKT over heads"
    else:
        title = (
            f"QKT mag | scale{stage_idx}_block{block_idx}_head_{head} "
            f"| stage{stage_idx:02d}_pn{patch_num} | call {call_idx}"
        )
        cbar_label = f"QKT head {head}"
    ax.set_title(title)
    ax.set_xlabel("key token index (Lk, cached)")
    ax.set_ylabel("query token index (Lq, this stage)")

    # Vertical lines separating cached scale segments on key axis.
    for b in cumulative_scale_bounds(patch_nums, stage_idx):
        ax.axvline(b - 0.5, color="white", lw=0.5, alpha=0.35)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_qkt_mag_heads_grid(
    qkt_mag: torch.Tensor,
    out_path: Path,
    stage_idx: int,
    patch_num: int,
    block_idx: int,
    patch_nums: Sequence[int],
    call_idx: int = 0,
    nrows: int = 4,
    ncols: int = 4,
) -> None:
    """Save all heads of one block as a 4x4 QKT magnitude grid."""
    assert qkt_mag.ndim == 3, f"expected [H,Lq,Lk], got {tuple(qkt_mag.shape)}"
    H = qkt_mag.shape[0]
    assert H == nrows * ncols, f"need {nrows*ncols} heads for {nrows}x{ncols} grid, got {H}"

    _ensure_dir(out_path)
    # Shared color scale across heads for fair comparison
    vmin = float(qkt_mag.min())
    vmax = float(qkt_mag.max())
    bounds = cumulative_scale_bounds(patch_nums, stage_idx)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), constrained_layout=True)
    im = None
    for h in range(H):
        r, c = divmod(h, ncols)
        ax = axes[r, c]
        mat = qkt_mag[h].numpy()
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        for b in bounds:
            ax.axvline(b - 0.5, color="white", lw=0.4, alpha=0.3)
        ax.set_title(f"head {h}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="QKT per head")
    fig.suptitle(
        f"QKT mag | scale{stage_idx}_block{block_idx} | stage{stage_idx:02d}_pn{patch_num} | call {call_idx}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_full_attention_matrix(
    attn: torch.Tensor,
    out_path: Path,
    title: str = "",
    head: Optional[int] = None,
    patch_nums: Optional[Sequence[int]] = None,
    stage_idx: Optional[int] = None,
) -> None:
    """Save classic Lq x Lk attention heatmap."""
    if attn.ndim == 3:
        mat = attn.mean(0).numpy() if head is None else attn[head].numpy()
        head_label = "mean" if head is None else str(head)
    else:
        mat = attn.numpy()
        head_label = "single"

    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, interpolation="nearest")
    ax.set_xlabel("Key token index")
    ax.set_ylabel("Query token index")
    ax.set_title(f"{title}\nhead={head_label}" if title else f"head={head_label}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if patch_nums is not None and stage_idx is not None:
        # Draw boundaries between cumulative scales present in keys.
        bounds = []
        cur = 0
        for si, pn in enumerate(patch_nums):
            cur += pn * pn
            if si < stage_idx:
                bounds.append(cur - 0.5)
        for b in bounds:
            ax.axvline(b, color="white", lw=0.6, alpha=0.5)
            ax.axhline(b, color="white", lw=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_query_key_heatmap(
    attn: torch.Tensor,
    out_path: Path,
    title: str = "",
    query_idx: int = -1,
    patch_nums: Optional[Sequence[int]] = None,
) -> None:
    """Save one query row as a bar-style heat strip + 16x16 spatial map side by side."""
    attn_2d = attn.mean(0) if attn.ndim == 3 else attn
    q = attn_2d.shape[0] + query_idx if query_idx < 0 else query_idx
    vec = attn_2d[q].numpy()

    _ensure_dir(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].imshow(vec[None, :], aspect="auto", cmap="hot", vmin=0.0)
    axes[0].set_title(f"Query token {q} → all keys")
    axes[0].set_xlabel("Key index")
    axes[0].set_yticks([])

    if patch_nums is not None:
        final_pn = patch_nums[-1]
        canvas = np.zeros((final_pn, final_pn), dtype=np.float32)
        counts = np.zeros_like(canvas)
        cur = 0
        for pn in patch_nums:
            chunk = vec[cur:cur + pn * pn]
            rs = final_pn // pn
            for idx, val in enumerate(chunk):
                r, c = idx // pn, idx % pn
                canvas[r * rs:(r + 1) * rs, c * rs:(c + 1) * rs] += val
                counts[r * rs:(r + 1) * rs, c * rs:(c + 1) * rs] += 1
            cur += pn * pn
        canvas = canvas / np.maximum(counts, 1)
        im = axes[1].imshow(canvas, cmap="hot")
        axes[1].set_title("Spatial attention (16×16 upscaled)")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].axis("off")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_panel(
    generated_path: Path,
    attn: torch.Tensor,
    out_path: Path,
    title: str,
    query_idx: int = -1,
    patch_nums: Optional[Sequence[int]] = None,
) -> None:
    """One-page summary: generated image + full matrix + spatial map."""
    from PIL import Image

    attn_2d = attn.mean(0).numpy() if attn.ndim == 3 else attn.numpy()
    q = attn_2d.shape[0] + query_idx if query_idx < 0 else query_idx

    _ensure_dir(out_path)
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(Image.open(generated_path))
    ax0.set_title("Generated")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(attn_2d, aspect="auto", cmap="viridis", vmin=0.0, interpolation="nearest")
    ax1.axhline(q, color="red", lw=0.8, alpha=0.8)
    ax1.set_title("Attention matrix (head mean)")
    ax1.set_xlabel("Key")
    ax1.set_ylabel("Query")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[2])
    if patch_nums is not None:
        vec = attn_2d[q]
        final_pn = patch_nums[-1]
        canvas = np.zeros((final_pn, final_pn), dtype=np.float32)
        counts = np.zeros_like(canvas)
        cur = 0
        for pn in patch_nums:
            chunk = vec[cur:cur + pn * pn]
            rs = final_pn // pn
            for idx, val in enumerate(chunk):
                r, c = idx // pn, idx % pn
                canvas[r * rs:(r + 1) * rs, c * rs:(c + 1) * rs] += val
                counts[r * rs:(r + 1) * rs, c * rs:(c + 1) * rs] += 1
            cur += pn * pn
        canvas = canvas / np.maximum(counts, 1)
        im2 = ax2.imshow(canvas, cmap="hot")
        ax2.set_title(f"Query {q} spatial map")
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pyramid_map(
    attn_vec: np.ndarray,
    patch_nums: Sequence[int],
    out_path: Path,
    title: str = "",
) -> None:
    """Visualize attention over the final 16x16 token grid."""
    final_pn = patch_nums[-1]
    canvas = np.zeros((final_pn, final_pn), dtype=np.float32)
    counts = np.zeros_like(canvas)

    cur = 0
    for pn in patch_nums:
        n = pn * pn
        chunk = attn_vec[cur:cur + n]
        for idx, val in enumerate(chunk):
            r = idx // pn
            c = idx % pn
            # Upscale low-res token to final grid cells.
            rs = final_pn // pn
            canvas[r * rs:(r + 1) * rs, c * rs:(c + 1) * rs] += val
            counts[r * rs:(r + 1) * rs, c * rs:(c + 1) * rs] += 1
        cur += n

    counts = np.maximum(counts, 1)
    canvas = canvas / counts

    _ensure_dir(out_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(canvas, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
