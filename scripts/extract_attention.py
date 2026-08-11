#!/usr/bin/env python3
"""Extract and visualize VAR attention maps during autoregressive inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attn_capture import AttentionCapture, run_infer_with_capture
from src.load_model import load_var_d16
from src.token_layout import build_token_layout
from src.visualize import save_qkt_mag_heads_grid, save_qkt_mag_heatmap


def parse_args():
    p = argparse.ArgumentParser(description="Extract VAR attention maps")
    p.add_argument("--class_id", type=int, default=980)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--blocks", type=str, default="0,8,15", help="Comma-separated block indices")
    p.add_argument("--stages", type=str, default="0,4,9", help="Comma-separated stage indices")
    p.add_argument("--query", type=int, default=-1, help="Query token index within current stage")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="4x4 grids + .pt (default: outputs/grids/class{class_id})",
    )
    p.add_argument("--cfg", type=float, default=1.0, help="Use 1.0 for clean single-branch attention")
    p.add_argument(
        "--save-per-head",
        action="store_true",
        help="Also save scaleX_blockX_head_H.png for each head (default: only 4x4 grid)",
    )
    p.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip the 4x4 grid image scaleX_blockX.png",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / (args.out_dir or f"outputs/grids/class{args.class_id}")
    extras = ROOT / "outputs" / "extras"
    gen_dir = extras / "generated"
    meta_dir = extras / "meta"
    head_root = extras / "per_head" / f"class{args.class_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    target_blocks = [int(x) for x in args.blocks.split(",") if x.strip()]
    target_stages = [int(x) for x in args.stages.split(",") if x.strip()]

    _, var, patch_nums = load_var_d16(depth=args.depth)
    tokens, stage_ranges = build_token_layout(patch_nums)

    capture = AttentionCapture()
    capture.install(var)

    image, stage_outputs = run_infer_with_capture(
        var,
        class_id=args.class_id,
        capture=capture,
        cfg=args.cfg,
        seed=args.seed,
        target_blocks=target_blocks,
    )

    # Save generated image for reference
    import torchvision

    torchvision.utils.save_image(image, gen_dir / f"class{args.class_id}.png")

    meta = {
        "class_id": args.class_id,
        "seed": args.seed,
        "patch_nums": list(patch_nums),
        "num_tokens": len(tokens),
        "stage_ranges": stage_ranges,
        "target_blocks": target_blocks,
        "target_stages": target_stages,
        "save_per_head": args.save_per_head,
        "save_grid": not args.no_grid,
    }
    (meta_dir / f"class{args.class_id}.json").write_text(json.dumps(meta, indent=2))

    for stage in stage_outputs:
        si = stage["stage_idx"]
        if si not in target_stages:
            continue
        for rec in stage["records"]:
            bi = rec["block_idx"]
            if bi not in target_blocks:
                continue
            attn = rec["attn"]  # H, Lq, Lk
            qkt = rec["qkt_mag"]
            pn = stage["patch_num"]
            num_heads = qkt.shape[0]
            scale_dir = out_dir / f"scale{si}"
            scale_dir.mkdir(parents=True, exist_ok=True)

            if args.save_per_head:
                head_dir = head_root / f"scale{si}"
                head_dir.mkdir(parents=True, exist_ok=True)
                for h in range(num_heads):
                    save_qkt_mag_heatmap(
                        qkt,
                        head_dir / f"scale{si}_block{bi}_head_{h}.png",
                        stage_idx=si,
                        patch_num=pn,
                        block_idx=bi,
                        patch_nums=patch_nums,
                        call_idx=0,
                        head=h,
                    )

            if not args.no_grid:
                save_qkt_mag_heads_grid(
                    qkt,
                    scale_dir / f"scale{si}_block{bi}.png",
                    stage_idx=si,
                    patch_num=pn,
                    block_idx=bi,
                    patch_nums=patch_nums,
                    call_idx=0,
                )

            torch.save(
                {
                    "stage_idx": si,
                    "block_idx": bi,
                    "qkt_mag": qkt,
                    "attn": attn,
                    "Lq": rec["Lq"],
                    "Lk": rec["Lk"],
                    "num_heads": num_heads,
                },
                scale_dir / f"scale{si}_block{bi}.pt",
            )

    print(f"Done. Grids -> {out_dir.resolve()}")
    print(f"       extras -> {extras.resolve()}")


if __name__ == "__main__":
    main()
