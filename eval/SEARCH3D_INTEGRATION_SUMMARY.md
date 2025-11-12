# Search3D Evaluation Integration Summary

**Date**: 2025-11-07
**Status**: Evaluation code integrated, GT download pending

---

## ✅ 完成項目

### 1. ✅ 評估代碼整合

**位置**: `eval/search3d/`

**複製的文件**:
```
eval/search3d/
├── eval_semantic_instance_parts_OV.py  ← 主要評估腳本
├── eval_semantic_instance_objects.py
├── eval_semantic_instance_parts.py
├── util.py
├── util_3d.py
└── data/
    ├── multiscan_search3d_constants.py  ← 標籤映射
    └── test_scenes_multiscan.txt        ← 41 個測試場景
```

**來源**: https://github.com/aycatakmaz/search3d

### 2. ✅ Quick Eval 更新

**修改**: `scripts/quick_eval_experiment.py`

**變更**:
- `run_search3d_evaluation()` 方法現在使用本地評估代碼
- 不再需要安裝 search3d package
- 直接從 `eval/search3d/` 導入

**測試驗證**:
```bash
python3 scripts/verify_gt_setup.py
```

結果：
- ✓ 評估代碼完整
- ✓ Constants 和 test scenes 文件存在
- ⚠ GT 數據需要下載（見下方）

### 3. ✅ 文檔與腳本

**創建的文件**:
1. `scripts/DOWNLOAD_GT_INSTRUCTIONS.md` - GT 下載指引
2. `scripts/verify_gt_setup.py` - 驗證設置腳本
3. `scripts/setup_search3d_gt.sh` - 自動設置腳本（可選）

---

## ⏳ 待完成項目

### GT 數據下載

**下載鏈接**: https://drive.google.com/drive/folders/1pyTHX3Ym8-StdWDvCVv72C6N_8GsjvSx?usp=sharing

**需要下載**:
- `multiscan_data_search3d/` 資料夾（完整）

**保存位置**:
```
/media/Pluto/richkung/My3DIS/data/search3d_gt/multiscan_data_search3d/
```

**預期結構**:
```
data/search3d_gt/
└── multiscan_data_search3d/
    ├── multiscan_annotations_search3d/
    │   └── ov_part_annotations/          ← 主要評估數據
    │       ├── scene_00005_00_obj_part_inst.txt
    │       ├── scene_00005_01_obj_part_inst.txt
    │       └── ... (41 files total)
    ├── multiscan_test_plys_only/
    │   └── *.ply (41 files)
    └── multiscan_search3d_constants.py
```

**下載指引**: 見 `scripts/DOWNLOAD_GT_INSTRUCTIONS.md`

---

## 📖 使用方式

### 驗證設置

```bash
python3 scripts/verify_gt_setup.py
```

### 運行評估（GT 下載完成後）

```bash
conda run -n 3Dsiglip python scripts/quick_eval_experiment.py \
    --exp-dir outputs/experiments/test_1105_ssam4_filter300_fill500_fix/scene_00005_00 \
    --dataset-root /media/public_dataset2/multiscan \
    --gt-path data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations \
    --output-dir eval_results \
    --config configs/inference/quick_eval.yaml
```

### 測試 Pipeline（無需 GT）

如果只想測試 pipeline 而不計算 mAP:

```bash
mkdir -p /tmp/dummy_gt

conda run -n 3Dsiglip python scripts/quick_eval_experiment.py \
    --exp-dir outputs/experiments/test_1105_ssam4_filter300_fill500_fix/scene_00005_00 \
    --dataset-root /media/public_dataset2/multiscan \
    --gt-path /tmp/dummy_gt \
    --output-dir eval_results
```

這會運行完整 pipeline 但跳過 mAP 計算。

---

## 🔍 技術細節

### 評估協議

**論文**: Search3D: Hierarchical Open-Vocabulary 3D Segmentation (RA-L 2025)

**任務**: Open-Vocabulary Part Segmentation
- 物體-部件對（例如: "cabinet door", "table drawer"）
- 47 個有效關係對
- 標籤格式: 6 位數字 `AAAIII`
  - `AAA`: 語義 ID
  - `III`: 實例編號

**指標**:
- mAP@25 (IoU threshold = 25%)
- mAP@50 (IoU threshold = 50%)
- mAP (平均 mAP)

### 測試場景

**總數**: 41 個場景文件
**唯一場景**: 20 個（scene_00005, scene_00006, ..., scene_00098）
**變體**: 大多數有 _00 和 _01，scene_00098 有 _00, _01, _02

**你當前的測試場景**: scene_00005_00 ✓ (包含在測試集中)

### 數據格式

**預測格式** (由 My3DIS inference pipeline 生成):
```python
predictions = {
    "scene_00005_00": {
        "pred_classes": np.array([...]),   # 類別 IDs
        "pred_scores": np.array([...]),    # 信心分數
        "pred_masks": np.array([...])      # Masks (N_points × N_instances)
    }
}
```

**GT 格式** (每個場景一個文件):
- 每行一個標籤
- 對應 PLY 文件中的點
- 標籤 0 = 背景
- 非零標籤 = `AAAIII` 格式

---

## 📊 評估 Pipeline 流程

```
1. Proposal Loading
   ↓
2. SigLIP Feature Extraction (真實 RGB)
   ↓
3. Inference (OV Part Segmentation)
   ↓
4. Prediction Formatting (Search3D 格式)
   ↓
5. Evaluation (mAP 計算)
   ├─ Load GT annotations
   ├─ Compute IoU overlaps
   ├─ Calculate AP per class
   └─ Average across classes
```

---

## ✅ 驗證清單

- [x] 評估代碼複製完成
- [x] Constants 和測試場景列表存在
- [x] Quick eval 腳本更新
- [x] 驗證腳本創建
- [x] 文檔完整
- [ ] GT 數據下載 ← **待完成**
- [ ] End-to-end 測試

---

## 🔗 參考資料

**Repository**: https://github.com/aycatakmaz/search3d

**文檔**:
- [MultiScan Data README](https://github.com/aycatakmaz/search3d/tree/main/search3d/benchmark/docs/multiscan_data_search3d)
- [Evaluation Code](https://github.com/aycatakmaz/search3d/tree/main/search3d/benchmark/evaluation/multiscan)

**Paper**:
```bibtex
@article{takmaz2025search3d,
  title={{Search3D: Hierarchical Open-Vocabulary 3D Segmentation}},
  author={Takmaz, Ayca and Delitzas, Alexandros and Sumner, Robert W. and
          Engelmann, Francis and Wald, Johanna and Tombari, Federico},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025}
}
```

---

## 🎯 下一步

1. **立即**: 從 Google Drive 下載 GT 數據
2. **驗證**: 運行 `python3 scripts/verify_gt_setup.py`
3. **測試**: 在 scene_00005_00 上運行完整評估
4. **擴展**: 批量評估全部 41 個場景

**預期耗時**:
- GT 下載: ~5-10 分鐘（取決於網速）
- 單場景評估: ~30 秒（含特徵提取）
- 全部場景: ~20 分鐘

---

**總結**: 評估架構已完全整合，只差 GT 數據下載即可開始真實評估！🎉
