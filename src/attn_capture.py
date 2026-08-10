from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class AttentionCapture:
    """Monkey-patch VAR SelfAttention to store softmax attention maps."""

    def __init__(self) -> None:
        self.enabled = False
        self.records: List[Dict] = []
        self._patched = False
        self._original_forward = None

    def clear(self) -> None:
        self.records.clear()

    def install(self, var_model) -> None:
        if self._patched:
            return
        from models.basic_var import SelfAttention

        self._original_forward = SelfAttention.forward
        capture = self

        def forward_with_capture(self_attn, x, attn_bias):
            if not capture.enabled:
                return capture._original_forward(self_attn, x, attn_bias)

            B, Lq, C = x.shape
            qkv = F.linear(
                input=x,
                weight=self_attn.mat_qkv.weight,
                bias=torch.cat((self_attn.q_bias, self_attn.zero_k_bias, self_attn.v_bias)),
            ).view(B, Lq, 3, self_attn.num_heads, self_attn.head_dim)

            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc

            if self_attn.attn_l2_norm:
                scale_mul = self_attn.scale_mul_1H11.clamp_max(self_attn.max_scale_mul).exp()
                q = F.normalize(q, dim=-1).mul(scale_mul)
                k = F.normalize(k, dim=-1)
                scale = 1.0
            else:
                scale = self_attn.scale

            if self_attn.caching:
                if self_attn.cached_k is None:
                    self_attn.cached_k = k
                    self_attn.cached_v = v
                else:
                    k = self_attn.cached_k = torch.cat((self_attn.cached_k, k), dim=2)
                    v = self_attn.cached_v = torch.cat((self_attn.cached_v, v), dim=2)

            attn_logits = torch.matmul(q, k.transpose(-2, -1)).mul(scale)
            qkt_mag = attn_logits  # before attn_bias and softmax
            if attn_bias is not None:
                attn_logits = attn_logits + attn_bias

            attn_probs = attn_logits.softmax(dim=-1)
            capture.records.append(
                {
                    "block_idx": self_attn.block_idx,
                    "qkt_mag": qkt_mag[0].detach().cpu(),   # H, Lq, Lk
                    "attn": attn_probs[0].detach().cpu(),    # H, Lq, Lk
                    "Lq": Lq,
                    "Lk": k.shape[2],
                }
            )

            dropout_p = self_attn.attn_drop if self_attn.training else 0.0
            if dropout_p > 0:
                attn_probs = F.dropout(attn_probs, p=dropout_p, training=self_attn.training)

            oup = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, Lq, C)
            return self_attn.proj_drop(self_attn.proj(oup))

        SelfAttention.forward = forward_with_capture
        self._patched = True

    def uninstall(self) -> None:
        if not self._patched or self._original_forward is None:
            return
        from models.basic_var import SelfAttention

        SelfAttention.forward = self._original_forward
        self._patched = False


@torch.no_grad()
def run_infer_with_capture(
    var,
    class_id: int,
    capture: AttentionCapture,
    cfg: float = 1.0,
    top_k: int = 900,
    top_p: float = 0.95,
    seed: int = 0,
    target_blocks: Optional[List[int]] = None,
):
    """Run VAR autoregressive inference and capture attention per stage/block."""
    from models.helpers import sample_with_top_k_top_p_

    device = next(var.parameters()).device
    B = 1
    label_B = torch.tensor([class_id], device=device, dtype=torch.long)

    if seed is not None:
        var.rng.manual_seed(seed)
    rng = var.rng

    # Match official inference, but allow cfg=1 to avoid duplicated batch during research.
    if cfg == 1.0:
        sos = cond_BD = var.class_emb(label_B)
        batch_mul = 1
    else:
        sos = cond_BD = var.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=var.num_classes)), dim=0)
        )
        batch_mul = 2

    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(batch_mul * B, var.first_l, -1)
    next_token_map = next_token_map + var.pos_start.expand(batch_mul * B, var.first_l, -1)
    next_token_map = next_token_map + lvl_pos[:, :var.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])

    for b in var.blocks:
        b.attn.kv_caching(True)

    stage_outputs = []
    capture.clear()
    capture.enabled = True

    for si, pn in enumerate(var.patch_nums):
        cur_L += pn * pn
        cond_BD_or_gss = var.shared_ada_lin(cond_BD)
        x = next_token_map

        for bi, b in enumerate(var.blocks):
            if target_blocks is not None and bi not in target_blocks:
                # Still run block, but temporarily disable recording.
                prev = capture.enabled
                capture.enabled = False
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                capture.enabled = prev
            else:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

        logits_BlV = var.get_logits(x, cond_BD)
        ratio = si / var.num_stages_minus_1
        t = cfg * ratio
        if batch_mul == 2:
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(
            logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
        )[:, :, 0]
        h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var.Cvae, pn, pn)
        f_hat, next_token_map = var.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(var.patch_nums), f_hat, h_BChw
        )

        stage_records = [r for r in capture.records]
        stage_outputs.append(
            {
                "stage_idx": si,
                "patch_num": pn,
                "seq_len": cur_L,
                "records": stage_records,
            }
        )
        capture.records.clear()

        if si != var.num_stages_minus_1:
            next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
            next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var.patch_nums[si + 1] ** 2]
            if batch_mul == 2:
                next_token_map = next_token_map.repeat(2, 1, 1)

    for b in var.blocks:
        b.attn.kv_caching(False)

    capture.enabled = False
    image = var.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
    return image, stage_outputs
