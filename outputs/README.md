# outputs 佈局

```
outputs/
├── grids/                 # 只放 scale*_block*.{png,pt}
│   ├── class980/scale0…9/
│   └── class437/scale9/
└── extras/
    ├── per_head/          # 單頭 QKT：scale*_block*_head_*.png
    ├── spatial/           # KEY.row×KEY.col 單幀與影片
    ├── generated/         # 該 class 生成圖
    ├── meta/              # json
    └── early_volcano/     # 最早一輪 exploratory 輸出
```

`grids/` 裡每個檔名都是 `scale{s}_block{b}.png`（4×4 head grid）或同名 `.pt`（`qkt_mag` / `attn` tensor）。
`.pt` 預設 gitignore。
