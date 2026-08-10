#!/usr/bin/env python3
"""Plot QKT heatmaps from a saved *.pt.

Default: only scaleX_blockX.png (4x4 grid).
Optional: --save-per-head for scaleX_blockX_head_H.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualize import save_qkt_mag_heads_grid, save_qkt_mag_heatmap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=str, required=True, help="Path to stage*_block*.pt")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--patch_nums", type=str, default="1,2,3,4,5,6,8,10,13,16")
    p.add_argument(
        "--save-per-head",
        action="store_true",
        help="Also save individual scaleX_blockX_head_H.png files",
    )
    p.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip the 4x4 grid image",
    )
    args = p.parse_args()

    pt_path = Path(args.pt)
    out_dir = Path(args.out_dir) if args.out_dir else pt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(pt_path, map_location="cpu")
    si = int(data["stage_idx"])
    bi = int(data["block_idx"])
    qkt = data["qkt_mag"]
    patch_nums = tuple(int(x) for x in args.patch_nums.split(","))
    pn = patch_nums[si]
    num_heads = qkt.shape[0]

    # Prefer out_dir/scaleX/; if out_dir already ends with scaleX, use it directly.
    if out_dir.name == f"scale{si}":
        scale_dir = out_dir
    else:
        scale_dir = out_dir / f"scale{si}"
    scale_dir.mkdir(parents=True, exist_ok=True)

    if args.save_per_head:
        for h in range(num_heads):
            out = scale_dir / f"scale{si}_block{bi}_head_{h}.png"
            save_qkt_mag_heatmap(
                qkt,
                out,
                stage_idx=si,
                patch_num=pn,
                block_idx=bi,
                patch_nums=patch_nums,
                call_idx=0,
                head=h,
            )
            print(f"saved {out}")

    if not args.no_grid:
        grid_out = scale_dir / f"scale{si}_block{bi}.png"
        save_qkt_mag_heads_grid(
            qkt,
            grid_out,
            stage_idx=si,
            patch_num=pn,
            block_idx=bi,
            patch_nums=patch_nums,
            call_idx=0,
        )
        print(f"saved {grid_out}")

    print(f"Done -> {scale_dir.resolve()}")


if __name__ == "__main__":
    main()
