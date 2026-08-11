# 研究記錄：VAR 殘差量化與 Scale9 Attention 結構

- **日期**：2026-08-11
- **模型**：官方 VAR-d16 + Multi-Scale VQVAE（`vae_ch160v4096z32.pth`）
- **設定**：`patch_nums = (1,2,3,4,5,6,8,10,13,16)`，L = 680；研究 attention 時 `cfg=1.0`
- **主要 class**：980（volcano）、437（lighthouse）；seed = 0
- **資料位置**：`VAR/attention_research/outputs/`

---

## 1. Discrete tokenizer 為何還能做 residual？

核心問題：VAR 每層預測的是 **token ID（整數）**，重建卻寫成類似 \(Z_{n+1}=\mathrm{Upsample}(Z_n)+\Delta Z_{n+1}\)。ID 不能當門牌號碼相加（5+1≠語意上的 6）。

### 1.1 加法發生在連續 latent，不是 token

VQVAE encoder 先把圖編成 `f ∈ R^{32×16×16}`。`VectorQuantizer2` 在這張 **連續特徵** 上做多尺度殘差：

```
f_hat = 0,  f_rest = f
for scale k in 0..9:
    把 f_rest 縮到 pn×pn
    每個位置最近鄰 → token id          # 離散
    h = φ(upsample(embedding[id]))     # 查表變回 R^32
    f_hat += h
    f_rest -= h
Decoder(f_hat) → RGB
```

對應程式：`VAR/VAR/models/quant.py` 的 `f_to_idxBl_or_fhat`、`get_next_autoregressive_input`；出圖是 `vae.fhat_to_img(f_hat)`。

### 1.2 Token 的角色

- **每個 scale 都立刻產生獨立的 token 序列**（長度 \(pn^2\)），不是 scale9 結束後才把 `f_hat` 量化成一張 token map。
- 全模型 **共用一本 codebook**：`nn.Embedding(4096, 32)`。scale0 的 id=5 和 scale9 的 id=5 是同一個 32 維向量；差別在空間格子與 upsample 後的頻率。
- `tokenID → 向量` 是 **查表**，不是 VAR transformer 另學的轉換。VQVAE 先訓、之後凍結；VAR 只學「下一個 scale 該抽哪個 id」。
- 32 維向量是學出來的實數錨點，不是整數格子。`embedding[5]` 與 `embedding[6]` 在幾何上不必相近。

### 1.3 生成時的資料流

每個 scale：Transformer 預測 id → `embedding(id)` → upsample + φ → **加進 `f_hat`** → 把目前的 `f_hat` downsample 當下一 scale 的條件。  
最終圖像只吃 **累加完的連續 `f_hat`**，Decoder 看不到 id。

先前較易搞錯的點：

| 誤解 | 實際 |
|---|---|
| 每層獨立 quantizer | 共用 codebook；獨立的是各層 token 格子 |
| 像素域「原圖減重建」 | latent 上 `f_rest -= h` |
| \(Z_{n+1}=\mathrm{Up}(Z_n)+\Delta Z\) 在各自 \(h_n\times w_n\) 遞增 | 訓練／推論預設把每一層都 upsample 到 **16×16 再累加** |
| 全部在 `f_hat` 上算完，最後才變 token | 每層先有離散 id，再查表加進 `f_hat` |

---

## 2. Attention 實驗設定

- 量的是 **softmax 前的 QKT mag**（`scale * Q @ K^T`），與論文風格 heatmap 一致。
- KV cache：stage \(k\) 的 Q 是 **當前 scale 的輸入 embedding**（尚未 sample 出這一層 token）；K 是 cache 裡 scale 0..k 的 key。
- Scale9：Q 256（16×16），K 680。Key 切段：

| key index | scale | \(pn\) |
|---|---|---|
| 0–0 | 0 | 1 |
| 1–4 | 1 | 2 |
| 5–13 | 2 | 3 |
| 14–29 | 3 | 4 |
| 30–54 | 4 | 5 |
| 55–90 | 5 | 6 |
| 91–154 | 6 | 8 |
| 155–254 | 7 | 10 |
| **255–423** | **8** | **13** |
| **424–679** | **9** | **16** |

4×4 grid：`outputs/grids/class980/scale{0-9}/`、`outputs/grids/class437/scale9/`。

---

## 3. Scale8 vs Scale9：同一段 key 0–423

資料：class 980；K 是同一份 cache，差在 Q（13×13 vs 16×16）。

- **多數 head**：對 0–423 的平均 QKT 曲線 cosine 很高（block1 多數 > 0.93）。對 **s0–s7（0–254）** 幾乎沒變（head2/12/13 ≈ 0.97）。
- **主要變化在 255–423（scale8）**：在 scale8 這是當前 self-attn，到 scale9 變成「上一層歷史」，分數與 softmax 質量明顯下降（例：head14 70% → 8%）。
- **Head14 例外**：0–423 整體 cosine 僅 0.19；進 scale9 後改去看 424–679，連舊 key 的打分方式都換了。

結論：多數 head 對更早歷史的偏好沿用上一 scale；變的是「剛從當前變成過去」的那一層。Softmax 質量仍可能因 scale9 多了 256 個自己的 key 而重新分配。

---

## 4. Scale9 / Block2 / Head 8 與 12

針對 4×4 裡差異大的兩顆頭細看（class 980 與 437 結構幾乎相同）。

單頭圖：

- `outputs/extras/per_head/class980/scale9/scale9_block2_head_{8,12}.png`
- `outputs/extras/per_head/class437/scale9/scale9_block2_head_{8,12}.png`

### Head 8：當前尺度為主，但散

- ~50% attn 在 sn=9，~18% 在 sn=8，~13–16% 在 sn=1（2×2）
- Scale9 對角只略高於非對角；argmax 空間距離 ~7–9 / 16
- 動態範圍小，沒有尖對角 → 把當前 16×16 當較散的全域 context，兼看粗輪廓

### Head 12：跨尺度空間對位（很尖）

- ~53% 在 sn=9，~14% 在 sn=8
- Scale9 對角 QKT ~6.5 vs 非對角 ~0.3；argmax 距離 **1.37 / 16**；top-10 集中度 **66%**
- 1D heatmap 上：scale8 一段有階梯斜線，scale9 有亮主對角，上/下三角有帶斜線的長方形磚
- 與 **block1 head12**（粗尺度 sink，過半質量砸在 2×2）角色不同

---

## 5. Head12 空間結構：1D 斜線從哪來

Scale9 的 256 token 是 row-major：`index = row * 16 + col`。  
因此 256×256 self-attn 看起來像 **16×16 塊 16×16 小磚**：

- 磚 `(qR, kR)` = 影像第 qR 列的 query 看第 kR 列的 key
- 磚內對角 = **同一 column**

這解釋了「主對角 + 上下三角長方形也有對角」：不是另一套 block attention，是 **2D 局部鄰居被 raster 切成積木**。

### 5.1 相對 query 的峰值

最強不在自己，而在 **上一列、右一欄** `(Δr=−1, Δc=+1)`。  
Q=50 → 空間位置 `(row=3, col=2)`，峰值在 `(2, 3)`。

單幀：`outputs/extras/spatial/class980/scale9_block2_head_12_q50_spatial.png`

### 5.2 不是「釘在畫面右上」、也不是臂寬 1 的細十字

把 256 個 Q 依 raster 接成影片後，黃斑 **跟著紅點（query）走**，說明中心是 **相對該 query 的右上鄰域**，不是 16×16 畫布的固定角落。

| 區域（softmax，佔全部 680 key） | class 980 head12 |
|---|---|
| query 附近實心 3×3 | 30.2% |
| query 附近實心 5×5 | 48.3% |
| 整條同一列 + 同一欄（臂寬 1 的全圖細十字） | 15.4% |
| 細十字上但離交叉點 >2 格 | 0.9% |

若把「5×5 對應十字」理解成 **同一扇窗裡臂寬 1 的 ＋（9 格）**：

- 中心在 query：十字 14.5% ≪ 方塊 48.3%（質量在四角）
- 中心在右上 `(r−1,c+1)`：十字 39.3% / 方塊 46.6%（84% 在十字上）

### 5.3 正確的「大十字」：臂長 16、臂寬 5 → 135 格

\(16\times5 + 16\times5 - 5\times5 = 135\)  
（橫條 16×5、豎條 16×5、交叉 5×5 只算一次。貼邊會裁切，平均約 127 格。）

同一顆 head12：

| | 佔全部 key | 相對 scale9 那 256 格 |
|---|---|---|
| 粗十字（中心在 query） | **53.0%** | **≈ 99.8%** |
| 十字外四個角落 | 0.1% | ~0 |
| query 附近實心 5×5（交叉口） | 48.3% | ~91% |
| 兩臂扣掉交叉口 | ~4.7% | — |

十字內 / 外平均 QKT：`+2.85` / `−2.10`。中心改到 `(r−1,c+1)` 幾乎一樣（52.9%）。

**結論（head12, scale9, block2）：**

1. 在當前 16×16 上，幾乎不看粗十字以外的四個角落。
2. 能量仍堆在交叉口附近（尤其相對 query 的右上），兩臂不是均勻發亮。
3. 1D heatmap 的斜線與矩形磚，是這個「寬 5 的 ＋」被 row-major 攤平後的樣子。

---

## 6. 空間 QKT 影片

每個 Q 一張 `KEY.row × KEY.col`（只取 scale9 的 256 key），Q=0…255 raster，8 fps，色階全片共用。紅圈＋ = query；白 x = 該幀峰值。

| class | 影片 |
|---|---|
| 980 | `outputs/extras/spatial/class980/scale9_block2_head_12_q_spatial.mp4` |
| 437 | `outputs/extras/spatial/class437/scale9_block2_head_12_q_spatial.mp4` |

兩 class 的局部右上 blob / 粗十字形狀高度相似，比較像 **head 的幾何先驗**，不是特定語意（火山 vs 燈塔）。

---

## 7. 產物清單（本日相關）

```
outputs/grids/class980/scale9/scale9_block2.{png,pt}
outputs/grids/class437/scale9/scale9_block2.{png,pt}
outputs/extras/per_head/class{980,437}/scale9/scale9_block2_head_{8,12}.png
outputs/extras/spatial/class{980,437}/scale9_block2_head_12_q50_spatial.png
outputs/extras/spatial/class{980,437}/scale9_block2_head_12_q_spatial.mp4
```

重畫單頭或空間圖時，`.pt` 需在本地；repo 不追蹤 tensor。

---

## 8. 尚未做 / 可接續

- 粗十字（臂寬 5）是否也出現在 **其他 block / 其他 head**，或只是 block2 head12
- Scale8 的 13×13 上是否同樣是「寬 5 的 ＋」對到對應空間位置
- 臂寬對 QKT（softmax 前）做掃描：寬 1、3、5、7 的 mass–cell 曲線
- Class 條件是否只調峰值位置、不改十字幾何
- 這顆 head 的相對偏移 `(−1,+1)` 是否來自 positional embedding / AdaLN，而不是資料
