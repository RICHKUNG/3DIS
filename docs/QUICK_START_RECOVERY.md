# 快速開始：v2 實驗關係救援

## 🎯 狀態

✅ **已完成並測試**
- 新實驗自動產生關係索引
- v2 實驗救援工具（支援舊格式）
- 批次處理腳本

## 🚀 使用方法

### 1. 單一實驗救援

```bash
PYTHONPATH=src python3 -m my3dis.recover_relations \
    --experiment-dir /media/Pluto/richkung/My3DIS/outputs/experiments/v2_135_ssam2_filter500_fill3k_prop30_iou06_ds03/scene_00005_00 \
    --levels "1,3,5" \
    --containment-threshold 0.95 \
    --mask-scale-ratio 1.0
```

**處理時間：**約 30-60 秒/scene（視 object 數量而定）

### 2. 批次救援所有 v2 實驗

#### 預覽

```bash
python scripts/batch_recover_v2_relations.py --dry-run
```

**輸出範例：**
```
Found 290 total run directories
Need recovery: 290 runs

=== Dry run: would process ===
  v2_135_*/scene_00005_00 (levels: [1, 3, 5])
  v2_135_*/scene_00005_01 (levels: [1, 3, 5])
  ...
```

#### 執行（單 GPU）

```bash
python scripts/batch_recover_v2_relations.py
```

**預計時間：**290 scenes × 45秒 ≈ 3.6 小時

### 3. ⚡ 雙 GPU 並行加速

#### 方法 A：手動分割（推薦）

```bash
# Terminal 1 (GPU 0) - 處理前半部
CUDA_VISIBLE_DEVICES=0 python scripts/batch_recover_v2_relations.py --limit 145

# Terminal 2 (GPU 1) - 處理後半部
# 需要修改腳本跳過前 145 個
```

#### 方法 B：使用 GNU Parallel

```bash
# 生成待處理清單
python scripts/batch_recover_v2_relations.py --dry-run | grep "would process" | awk '{print $1}' > /tmp/scenes_to_recover.txt

# 並行處理（雙 GPU）
cat /tmp/scenes_to_recover.txt | parallel -j 2 --colsep ' ' \
    'CUDA_VISIBLE_DEVICES=$((GPU_ID++ % 2)) PYTHONPATH=src python3 -m my3dis.recover_relations \
     --experiment-dir {1} --levels {2} --containment-threshold 0.95 --mask-scale-ratio 1.0'
```

**預計時間：**290 ÷ 2 ≈ **1.8 小時**

## 📊 輸出檔案

每個scene會產生：

```
scene_*/
├── level_1/
│   ├── relations/tree.json     ← 新增
│   └── index.json               ← 新增
├── level_3/
│   ├── relations/tree.json     ← 新增
│   └── index.json               ← 新增
├── level_5/
│   ├── relations/tree.json     ← 新增
│   └── index.json               ← 新增
└── relations.json               ← 新增（跨 level 關係）
```

## 🔍 驗證

檢查單一實驗：

```bash
# 查看 relations.json
cat /path/to/scene_00005_00/relations.json | jq .meta

# 查看 index.json
cat /path/to/scene_00005_00/level_1/index.json | jq '.meta, .objects | keys | length'
```

預期輸出：

```json
{
  "scene": "scene_00005_00",
  "levels": [1, 3, 5],
  "generated_at": "2025-10-22T..."
}

166  // level 1 有 166 個 objects
```

## ⚙️ 參數調整

### Containment Threshold

```bash
--containment-threshold 0.90  # 更寬鬆（可能增加 false positives）
--containment-threshold 0.98  # 更嚴格（可能遺漏部分關係）
```

**建議：** 保持 0.95（經驗最佳值）

### Mask Scale Ratio

v2 實驗大多使用 **scale=1.0**（未縮放），如果看到警告 "No video_segments found"，請確認：

```bash
# 檢查實際檔名
ls level_1/video_segments*.npz
# 如果是 video_segments_L01.npz → 使用 --mask-scale-ratio 1.0
```

## 🐛 常見問題

### Q: 找不到 video_segments

```
WARNING: No video_segments found in .../level_1 or .../level_1/tracking
```

**解決：** 檢查檔案是否存在：
```bash
find /path/to/scene_00005_00 -name "video_segments*.npz"
```

### Q: 處理很慢

**原因：** Mask 包含關係計算需要載入所有 frames 的 masks

**加速方案：**
1. 使用 SSD 存儲實驗結果
2. 使用雙 GPU 並行
3. 減少 containment 檢查的 frames（需修改程式碼）

### Q: 記憶體不足

**調整：** 單一 scene 最多使用 ~8GB RAM（視 object 數量而定）

**若出現 OOM：**
```bash
# 限制並行數
parallel -j 1 ...  # 改用單執行緒
```

## 📈 進度監控

批次處理時查看進度：

```bash
# 查看已完成的 scenes
find /media/Pluto/richkung/My3DIS/outputs/experiments/v2_* -name "relations.json" | wc -l

# 查看當前處理的 scene（監控日誌）
tail -f /tmp/recovery_batch.log
```

## ✅ 完成後

所有 v2 實驗救援完成後，新舊實驗的資料格式會統一：

- ✅ Object ID ↔ Frames 映射
- ✅ Frame → Objects 映射
- ✅ Parent-Child 階層關係
- ✅ Ancestor Paths

可以開始進行下游分析！

---

**最後更新：** 2025-10-22
**作者：** Rich Kung with Claude Code
