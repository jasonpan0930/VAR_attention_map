#!/usr/bin/env python3
"""200-image LS-scale a std for s8→s9 s0–s7 (upsample 169×255 → 256×255).

Only heads with cosine > 0.9 contribute to per-head std.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attn_capture import AttentionCapture
from src.load_model import load_var_d16

NK = 255  # s0–s7 keys
COS_THR = 0.9


@torch.no_grad()
def run_s89(var, class_id: int, capture: AttentionCapture, seed: int = 0):
    from models.helpers import sample_with_top_k_top_p_

    device = next(var.parameters()).device
    B = 1
    label_B = torch.tensor([class_id], device=device, dtype=torch.long)
    var.rng.manual_seed(seed)
    rng = var.rng
    sos = cond_BD = var.class_emb(label_B)
    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    nxt = sos.unsqueeze(1).expand(B, var.first_l, -1)
    nxt = nxt + var.pos_start.expand(B, var.first_l, -1) + lvl_pos[:, : var.first_l]
    cur_L = 0
    f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])
    for b in var.blocks:
        b.attn.kv_caching(True)
    outs = {}
    capture.clear()
    for si, pn in enumerate(var.patch_nums):
        cur_L += pn * pn
        cond = var.shared_ada_lin(cond_BD)
        x = nxt
        capture.enabled = si in (8, 9)
        for b in var.blocks:
            x = b(x=x, cond_BD=cond, attn_bias=None)
        logits = var.get_logits(x, cond_BD)
        idx = sample_with_top_k_top_p_(logits, rng=rng, top_k=900, top_p=0.95, num_samples=1)[:, :, 0]
        h = var.vae_quant_proxy[0].embedding(idx).transpose(1, 2).reshape(B, var.Cvae, pn, pn)
        f_hat, nxt = var.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(var.patch_nums), f_hat, h
        )
        if si in (8, 9):
            outs[si] = {r["block_idx"]: r["qkt_mag"].float() for r in capture.records}
        capture.records.clear()
        if si != var.num_stages_minus_1:
            nxt = nxt.view(B, var.Cvae, -1).transpose(1, 2)
            nxt = var.word_embed(nxt) + lvl_pos[:, cur_L : cur_L + var.patch_nums[si + 1] ** 2]
    for b in var.blocks:
        b.attn.kv_caching(False)
    capture.enabled = False
    return outs


def upsample_q(a8: torch.Tensor) -> torch.Tensor:
    H, Lq, Lk = a8.shape
    assert Lq == 169 and Lk == NK, (Lq, Lk)
    x = a8.view(H, 13, 13, Lk).permute(0, 3, 1, 2).contiguous()
    up = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
    return up.permute(0, 2, 3, 1).reshape(H, 256, Lk)


def main():
    class_ids = list(range(0, 1000, 5))  # 200 classes: 0,5,...,995 (includes 980)
    assert len(class_ids) == 200
    if 437 not in class_ids:
        class_ids[class_ids.index(435)] = 437
    jobs = [(cid, 0) for cid in class_ids]
    n_img = len(jobs)

    meta = ROOT / "outputs/extras/meta"
    spatial = ROOT / "outputs/extras/spatial/class980"
    meta.mkdir(parents=True, exist_ok=True)
    spatial.mkdir(parents=True, exist_ok=True)

    print("load", flush=True)
    _, var, _ = load_var_d16(depth=16)
    device = next(var.parameters()).device
    print(f"device={device}  n_images={n_img}", flush=True)
    cap = AttentionCapture()
    cap.install(var)

    long_rows = []
    for i, (cid, seed) in enumerate(jobs, 1):
        print(f"[{i:3d}/{n_img}] infer c{cid} s{seed}", flush=True)
        outs = run_s89(var, cid, cap, seed=seed)
        for bi in range(16):
            a8 = outs[8][bi][:, :, :NK]
            a9 = outs[9][bi][:, :, :NK]
            u8 = upsample_q(a8)
            H = a8.shape[0]
            v8 = u8.reshape(H, -1)
            v9 = a9.reshape(H, -1)
            cos = F.cosine_similarity(v8, v9, dim=-1)
            a_ls = (v8 * v9).sum(-1) / v8.square().sum(-1).clamp_min(1e-8)
            m8 = u8.mean(dim=(1, 2))
            m9 = a9.mean(dim=(1, 2))
            for h in range(H):
                r8, r9 = float(m8[h]), float(m9[h])
                long_rows.append(
                    {
                        "class_id": cid,
                        "seed": seed,
                        "block": bi,
                        "head": h,
                        "cosine": float(cos[h]),
                        "ls_scale": float(a_ls[h]),
                        "ratio": r9 / r8 if abs(r8) > 1e-6 else float("nan"),
                        "mean8_up": r8,
                        "mean9": r9,
                    }
                )
        del outs

    fields = ["class_id", "seed", "block", "head", "cosine", "ls_scale", "ratio", "mean8_up", "mean9"]
    long_path = meta / "upsample256_ls_scale_across_images_n200.csv"
    with long_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in long_rows:
            w.writerow(
                {
                    k: (f"{r[k]:.6f}" if k not in ("class_id", "seed", "block", "head") else r[k])
                    for k in fields
                }
            )

    g_hi = defaultdict(list)
    n_hi = 0
    for r in long_rows:
        if r["cosine"] > COS_THR:
            g_hi[(r["block"], r["head"])].append(r)
            n_hi += 1

    sum_rows = []
    for bi in range(16):
        for h in range(16):
            xs = g_hi.get((bi, h), [])
            if len(xs) < 2:
                continue
            als = [x["ls_scale"] for x in xs]
            rs = [x["ratio"] for x in xs]
            sum_rows.append(
                dict(
                    block=bi,
                    head=h,
                    n=len(xs),
                    n_images=n_img,
                    frac=len(xs) / n_img,
                    cosine_mean=st.mean([x["cosine"] for x in xs]),
                    a_mean=st.mean(als),
                    a_std=st.stdev(als),
                    a_min=min(als),
                    a_max=max(als),
                    a_range=max(als) - min(als),
                    ratio_mean=st.mean(rs),
                    ratio_std=st.stdev(rs),
                )
            )

    sum_path = meta / "upsample256_ls_scale_across_images_cosine_gt0.9_n200.csv"
    sfields = [
        "block",
        "head",
        "n",
        "n_images",
        "frac",
        "cosine_mean",
        "a_mean",
        "a_std",
        "a_min",
        "a_max",
        "a_range",
        "ratio_mean",
        "ratio_std",
    ]
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for r in sum_rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in sfields})

    print(f"\n--- {n_img} images, s0–s7 keys (255), cosine>{COS_THR} ---")
    print(f"rows cosine>0.9: {n_hi}/{n_img * 256} ({n_hi / (n_img * 256):.1%})")
    print(f"heads ≥2 imgs cosine>0.9: {len(sum_rows)}/256")
    if sum_rows:
        astds = [r["a_std"] for r in sum_rows]
        full = [r for r in sum_rows if r["n"] >= 0.8 * n_img]
        print(
            f"LS a std: mean={st.mean(astds):.4f} median={st.median(astds):.4f} "
            f"min={min(astds):.4f} max={max(astds):.4f}"
        )
        print(
            f"  std<0.05 {sum(s < 0.05 for s in astds) / len(astds):.1%}  "
            f"<0.10 {sum(s < 0.10 for s in astds) / len(astds):.1%}"
        )
        print(f"≥80% images cosine>0.9: {len(full)}")
        if full:
            as2 = [r["a_std"] for r in full]
            print(f"  a std mean={st.mean(as2):.4f} median={st.median(as2):.4f}")
            print(f"  std<0.05 {sum(s < 0.05 for s in as2) / len(as2):.1%}")
        print("most stable a:")
        for r in sorted(sum_rows, key=lambda x: x["a_std"])[:5]:
            print(
                f"  b{r['block']:02d} h{r['head']:02d} n={r['n']:3d}/{n_img} "
                f"a={r['a_mean']:.3f}±{r['a_std']:.3f}"
            )
        print("least stable a:")
        for r in sorted(sum_rows, key=lambda x: -x["a_std"])[:5]:
            print(
                f"  b{r['block']:02d} h{r['head']:02d} n={r['n']:3d}/{n_img} "
                f"a={r['a_mean']:.3f}±{r['a_std']:.3f}"
            )

    mat = np.full((16, 16), np.nan)
    for r in sum_rows:
        mat[r["block"], r["head"]] = r["a_std"]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    cmap = plt.cm.magma.copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(mat, origin="upper", cmap=cmap, vmin=0, vmax=0.08)
    ax.set_xlabel("head")
    ax.set_ylabel("block")
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.set_title(
        f"LS scale a std across {n_img} images / upsample 256×255, only cosine>0.9"
    )
    for i in range(16):
        for j in range(16):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="#888")
            else:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if v > 0.04 else "yellow",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="std of a")
    fig.tight_layout()
    figp = spatial / "upsample256_ls_scale_std_cosine_gt0.9_n200.png"
    fig.savefig(figp, dpi=180, bbox_inches="tight")
    plt.close()
    print("saved", long_path)
    print("saved", sum_path)
    print("saved", figp)


if __name__ == "__main__":
    main()
