from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class TokenInfo:
    index: int
    scale_idx: int
    patch_num: int
    row: int
    col: int
    label: str


def build_token_layout(patch_nums: Sequence[int]) -> Tuple[List[TokenInfo], List[Tuple[int, int]]]:
    """Map flat token indices to VAR pyramid scales.

    Returns:
        tokens: metadata for each token position in the full sequence (length L)
        stage_ranges: list of (start, end) token spans per autoregressive stage
    """
    tokens: List[TokenInfo] = []
    stage_ranges: List[Tuple[int, int]] = []
    cur = 0
    for si, pn in enumerate(patch_nums):
        start = cur
        for r in range(pn):
            for c in range(pn):
                tokens.append(
                    TokenInfo(
                        index=cur,
                        scale_idx=si,
                        patch_num=pn,
                        row=r,
                        col=c,
                        label=f"s{si}_p{pn}_({r},{c})",
                    )
                )
                cur += 1
        stage_ranges.append((start, cur))
    return tokens, stage_ranges


def stage_name(si: int, patch_nums: Sequence[int]) -> str:
    return f"stage{si}_p{patch_nums[si]}x{patch_nums[si]}"
