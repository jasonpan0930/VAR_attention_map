from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

# Reuse the official VAR codebase without modifying it.
VAR_ROOT = Path(__file__).resolve().parents[2] / "VAR"
if str(VAR_ROOT) not in sys.path:
    sys.path.insert(0, str(VAR_ROOT))

from models import build_vae_var  # noqa: E402


def load_var_d16(
    device: str | None = None,
    depth: int = 16,
    checkpoint_dir: Path | None = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, tuple]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = checkpoint_dir or (VAR_ROOT / "checkpoints")
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=depth, shared_aln=False,
        flash_if_available=False, fused_if_available=False,
    )

    vae_ckpt = ckpt_dir / "vae_ch160v4096z32.pth"
    var_ckpt = ckpt_dir / f"var_d{depth}.pth"
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval().to(device)
    var.eval().to(device)
    return vae, var, patch_nums
