# VAR Attention Map Research

獨立研究資料夾，用來分析 **VAR** 在 autoregressive inference 時的 self-attention。

## 資料夾結構

```
attention_research/
├── README.md
├── requirements.txt
├── src/
│   ├── load_model.py      # 載入官方 VAR-d16 + VAE
│   ├── token_layout.py    # token index ↔ scale/patch 對照
│   ├── attn_capture.py    # monkey-patch SelfAttention，存 softmax map
│   └── visualize.py       # heatmap / head grid / pyramid map
├── scripts/
│   └── extract_attention.py
├── outputs/               # 輸出 attention 圖和 .pt
└── notebooks/             # 可自行放 exploratory notebook
```

## 依賴

```bash
conda activate varsr
pip install matplotlib
```

VAR 本體與權重沿用 `../VAR/checkpoints/`（需已下載 `var_d16.pth` 和 `vae_ch160v4096z32.pth`）。

## 快速開始

```bash
cd /home/jason/courses/2026_spring/digital_research/VAR/attention_research
python scripts/extract_attention.py --class_id 980 --out_dir outputs/volcano
```

常用參數：

| 參數 | 說明 |
|------|------|
| `--class_id` | ImageNet class id |
| `--blocks` | 要記錄的 transformer block，例如 `0,8,15` |
| `--stages` | 要輸出的 autoregressive stage，例如 `0,4,9` |
| `--cfg` | 建議研究 attention 時用 `1.0`（不做 CFG 雙分支） |
| `--save-per-head` | 額外輸出每張 head 的單圖 `scaleX_blockX_head_H.png`（預設關閉） |
| `--no-grid` | 不輸出 4×4 總覽圖 |

## 輸出說明

預設每次執行會產生：

- `generated.png`：對應 class 的生成圖
- `meta.json`：patch pyramid / token range 資訊
- `scaleX_blockX.png`：該 block 的 **4×4 QKT head grid**
- `stage*_block*.pt`：原始 tensor（含 `qkt_mag` `[H, Lq, Lk]`）

若加 `--save-per-head`，另存：

- `scaleX_blockX_head_H.png`：單一 head 的 QKT heatmap

## VAR token 排列

`patch_nums = (1,2,3,4,5,6,8,10,13,16)`，總 token 數 L = 680。

Inference 是 **next-scale prediction**：
- stage 0 只有 1×1 token
- stage 9 有 16×16 = 256 tokens
- kv cache 會讓後面 stage 的 key 長度逐步累積

可用 `src/token_layout.py` 查 index 對應到哪個 scale。

## 實作備註

- 為了拿到 attention weight，會 **關閉 flash/xformers**，走 explicit softmax 路徑
- 使用 monkey-patch，不修改官方 `VAR/models/` 原始碼
- 研究時建議 `--cfg 1.0`，避免 CFG 複製 batch 讓 attention 解讀變複雜

## 下一步你可以做

- 比較不同 block / stage 的 attention pattern
- 看 class token (stage0) 對後續 scale 的影響
- 對照 FPQVAR 量化前後 attention 差異（可在此資料夾擴充）
