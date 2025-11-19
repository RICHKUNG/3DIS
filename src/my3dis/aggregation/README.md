# 3D Mask Aggregation 模組說明

本模組實作 3D 投影管線的第一階段：  
將 SAM2 追蹤得到的 2D instance masks 投影到 3D 點雲上，並在多幀之間聚合，產生穩定的 3D proposals。

---

## 整體概觀

**輸入**：SAM2 追蹤階段輸出的 `family_tree.json`（或 legacy `relations.json`）
**輸出**：各層級的 3D proposal NPZ 檔案（`proposal_data_level{L}.npz`，如 L1/L3/L5 或 L2/L4/L6），以及聚合 metadata。

**支援的層級配置**：
- 標準配置：L1/L3/L5（常用於MultiScan數據）
- 替代配置：L2/L4/L6 或其他自定義層級組合
- 層級自動檢測：由 `detect_available_levels()` 根據SAM2輸出動態決定

### 管線流程（概念）

```text
family_tree.json（SAM2 輸出）
    ↓
讀取各層級物件與對應影格
    ↓
對每一個 object：
  - 從所有影格載入 2D masks
  - 利用深度圖與相機參數投影到 3D
  - 在多幀之間進行聚合（union / intersection / majority）
    ↓
以體積門檻過濾小 proposal
    ↓
以 3D IoU 進行 deduplication
    ↓
輸出 3D proposals（NPZ）
```

---

## 模組結構

```text
aggregation/
├── __init__.py                 # 對外匯出 AggregationPipeline 等
├── aggregation_pipeline.py     # 主要管線 orchestration
├── mask_projection.py          # 2D → 3D 投影
├── mask_projection_3d_to_2d.py # 3D → 2D 回投影（視覺化 / 檢查用）
├── aggregation_utils.py        # 聚合與幾何工具
├── pipeline_functions.py       # 由 masklets 建構 proposals 的細部操作
├── proposal_loader.py          # 從 experiment 目錄載入 proposals
├── search3d_formatter.py       # Search3D 匯出與驗證
└── README.md                   # 本文件
```

---

## 主要元件與函式

### `aggregation_pipeline.py`

- `detect_available_levels(run_dir)`  
  探測 SAM2 run 目錄中哪些層級擁有完整輸出（`video_segments`/`object_segments` 等），供 CLI 在未指定 `--levels` 時自動決定處理層級。

- `_is_family_tree_format(payload)`／`_family_tree_to_relations(tree)`／`_relations_to_family_tree(relations)`／`_prepare_relations_data(path)`  
  處理 **新舊關聯格式** 的轉換：
  - 若已有 `family_tree.json`，直接使用；
  - 若只有 `relations.json`，以 `relations_to_family_tree` 轉為 family‑tree 風格；
  - 此統一介面讓後續程式不必關心來源版本差異。

- `AggregationPipeline` 類別  
  建立整個 3D 聚合流程的主 orchestrator。

  - `__init__(family_tree_path, pointcloud_path, depth_maps_dir, color_frames_dir, camera_params_path=None, iou_threshold=0.6, min_volume=0.001, vis_depth_threshold=0.05, device='cuda:0')`：
    - 讀取 family tree 或 relations；
    - 建立點雲與深度圖路徑設定；
    - 初始化 Mask projector、3D 聚合工具與相關閾值。
  - `_init_world2cam(camera_params_path)`：
    - 載入相機內外參資訊，建立 world→camera 的轉換矩陣；
    - 當 camera params 缺失時，會提供合理錯誤訊息。
  - `run(output_dir, levels=None)`：
    - 針對指定（或自動偵測）的各層級：
      1. 從 family tree 查詢每個物件的 frame 列表；
      2. 透過 `MaskProjector.project_mask_to_3d` 將每幀 mask 投影到點雲索引；
      3. 以 `aggregate_masks_3d` 聚合多幀遮罩；
      4. 以 `filter_proposals_by_volume` 過濾體積過小 proposal；
      5. 以 `compute_iou_3d` 做 proposal 間 dedup；
      6. 將結果寫入 `proposal_data_level{L}.npz`；
    - 同時產生 `aggregation_metadata.json`，紀錄參數與統計；
    - 回傳 `{level_str: npz_path}` 的字典。

- `main()`  
  對應 CLI：

```bash
PYTHONPATH=src python -m my3dis.run_aggregation \
  --family-tree path/to/family_tree.json \
  --pointcloud  path/to/scene/pointcloud.ply \
  --depth-maps-dir  path/to/scene/depth \
  --color-frames-dir path/to/scene/color \
  --output-dir outputs/aggregation/scene_00065_00 \
  --levels 1 3 5 \  # 或使用 2 4 6，視數據而定
  --iou-threshold 0.6 \
  --min-volume 0.001 \
  --device cuda:0
```

---

### `mask_projection.py` 與 `mask_projection_3d_to_2d.py`

#### `mask_projection.py`

- `MaskProjector` 類別
  - `__init__(pointcloud_path, depth_maps_dir, color_frames_dir, camera_params_path, vis_depth_threshold)`：
    - 載入點雲與相機參數；
    - 設定深度一致性判斷閾值（`vis_depth_threshold`）。
  - `_load_pointcloud()`／`_load_ply()`：
    - 支援 `.ply` 與部分二進位格式（視實作而定）；
    - 讀取點座標與顏色（若有）。
  - `_load_camera_params()`：
    - 從 JSON 或其他格式讀入相機內外參；
    - 建立 per‑frame 的投影矩陣。
  - `_load_depth_map(frame)`／`_read_depth_file(path)`：
    - 支援 PNG／EXR 等深度格式；
    - 處理單位轉換（例如 mm → meters）。
  - `_project_points_to_2d(points, camera)`：
    - 以內外參與深度資訊將 3D 點投影至 2D 平面；
    - 判斷可見性與遮擋情況。
  - `project_mask_to_3d(object_masks, frames)`：
    - 將 2D instance masks 投影到 3D點雲，建立 `N_points × N_objects` 的二值遮罩。

#### `mask_projection_3d_to_2d.py`

- `Mask3Dto2DProjector` 類別
  - 反向操作：給定 3D proposal mask，將其投影回 2D 影像；
  - `project_3d_mask_to_2d(...)`：在某幀中標示該 3D proposal 投影到畫面的像素位置；
  - `get_visible_frames(...)`：找出哪些 frame 可以看到此 3D proposal，常用於可視化或診斷。
- `main()`：提供命令列工具，用於單獨測試投影與回投影效果。

---

### `aggregation_utils.py`

- `aggregate_masks_3d(mask_stack, method='union')`：
  - 針對同一個物件在多幀的 3D masks 進行聚合：
    - `union`：只要在任一幀可見即納入；
    - `intersection`：所有幀都可見才納入；
    - `majority`：在超過半數幀可見才納入。
- `compute_proposal_geometry(points)`：
  - 根據 3D 點集計算：
    - 質心（centroid）；
    - 軸對齊 bounding box；
    - 近似體積與表面積。
- `filter_proposals_by_volume(proposals, min_volume)`：
  - 過濾體積低於閾值的 proposal；
  - 回傳保留下來的 proposals 與對應索引。
- `compute_iou_3d(mask_a, mask_b)`：
  - 在 3D 空間中計算兩個 proposal 之 IoU；
  - 用於 dedup 與 NMS。

---

### `pipeline_functions.py`

該模組集中與「由 masklet 重建 proposal」相關的底層運算。

- `build_proposal_masks_from_masklets(masklets, level_cfg)`：
  - 從 SAM2 masklets（通常為 frame x instance 格式）重建 per‑object proposal masks；
  - 可能需要先做 IoU 群聚與 parent grouping。
- 輔助函式：
  - `_areas_from_mask(mask)`：計算遮罩面積；
  - `_iou_matrix(masks)`：建立所有 mask 間的 IoU 矩陣；
  - `_connected_components_by_thresh(iou_mat, thresh)`：
    - 根據 IoU threshold 把高相似的 masks 分群；
  - `_parent_groups_for_level(...)`／`_ids_to_cols(...)`／`_safe_get_level(...)`：
    - 處理層級與 id 的轉換與查詢。
- 變形與調整：
  - `merge_drop_level2_mask(...)`：
    - 在 L2 層級上合併／刪除不穩定或碎裂的 masks；
  - `split_merged_vs_small(...)`：
    - 將 merged proposals 與過小 proposals 分開，以便針對兩者採用不同策略；
  - `update_relations_level2_inplace_v2(...)`：
    - 配合 L2 合併／刪除操作，直接修正 relations tree 中的 parent-child 關係；
  - `merge_drop_level4_within_siblings`／`merge_drop_level6_within_siblings`：
    - 在兄弟節點之間進行合併與刪除，減少冗餘。
- 特徵池化：
  - `pool_point_to_proposal_features(points, masklets, features)`：
    - 將點層級特徵（例如 SigLIP）聚合為 proposal 級別向量（例如平均或加權平均）；
    - 供後續 SigLIP assignment 或推論使用。

---

### `proposal_loader.py`

- `ProposalLoader` 類別：
  - `__init__(experiment_root)`：
    - 預先記錄 experiment 根目錄，方便尋找特定場景／層級的 proposal。
  - `get_scene_name()`：
    - 從目錄結構推斷場景名稱（例如 `scene_00065_00`）。
  - `_find_level_dir(level)`／`_find_tracking_files(level_dir)`：
    - 尋找各層級的輸出位置與對應追蹤檔案。
  - `_load_level_proposals(level_dir)`：
    - 讀取 `proposal_data_level{L}.npz`（或等價檔案），建立方便操作之 in‑memory 結構。
  - `_aggregate_masks(...)`：
    - 若需要，將 masklets 重新聚合為 per‑proposal 遮罩。
- 高階函式：
  - `load_proposals_from_experiment(experiment_root, levels)`：
    - 由外部腳本呼叫，回傳 `{level_str: level_proposals}`；
    - 供 evaluation 或分析使用。

---

### `search3d_formatter.py`

- `export_predictions_search3d(predictions, output_path)`：
  - 將 `Inference` 模組輸出的 `Prediction` 集合轉換成 Search3D 所要求的矩陣格式（`pred_classes`／`pred_scores`／`pred_masks`），並寫成 `.npz`。
- `export_predictions_both_formats(predictions, output_dir)`：
  - 同時輸出 Search3D 格式與自訂 JSON/npz 格式，便於內部調試與外部提交。
- `load_point_cloud_for_export(scene_dir)`：
  - 由 scene 目錄載入點雲資料，以便在輸出時確認點數與遮罩 shape 是否一致。
- `validate_search3d_format(pred_payload)`：
  - 檢查輸出 payload：
    - 形狀是否一致（`pred_masks.shape[1] == pred_classes.shape[0] == pred_scores.shape[0]`）；
    - dtype 是否符合規範（classes:int32, scores:float, masks:uint8）。

---

## 輸出檔案格式

### `proposal_data_level{L}.npz`

每一層級的 NPZ 大致包含：

```python
{
    'masks':    np.ndarray,  # [N_points, N_proposals]  binary masks
    'obj_ids':  np.ndarray,  # [N_proposals]           object IDs (family tree)
    'levels':   np.ndarray,  # [N_proposals]           level numbers (全部為該 L)
    'parent_ids': np.ndarray,# [N_proposals]           parent IDs (-1 代表 root)
    'volumes':  np.ndarray,  # [N_proposals]           proposal 體積 (m^3)
    'centroids': np.ndarray, # [N_proposals, 3]        質心
    'bboxes':    np.ndarray, # [N_proposals, 6]        軸對齊 bbox (min_xyz, max_xyz)
    'points':    np.ndarray, # [N_points, 3]           點雲座標
    'colors':    np.ndarray, # [N_points, 3]           點雲顏色（若有）
}
```

### `aggregation_metadata.json`

```json
{
  "family_tree": "path/to/family_tree.json",
  "pointcloud":  "path/to/pointcloud.ply",
  "iou_threshold": 0.6,
  "min_volume": 0.001,
  "outputs": {
    "1": "proposal_data_level1.npz",  // 或 "2" 視層級配置而定
    "3": "proposal_data_level3.npz",  // 或 "4"
    "5": "proposal_data_level5.npz"   // 或 "6"
  },
  "statistics": {
    "total_objects": 150,
    "total_families": 45
  }
}
```

---

## 與 Family Tree 與後續階段之整合

本模組直接消費 SAM2 階段的 `family_tree.json`：

```python
from my3dis.family_tree_query import FamilyTreeQuery

query = FamilyTreeQuery('relations/family_tree.json')

# 取得 L2 層級的所有物件
obj_ids = query.search_objects_by_level(2)

for obj_id in obj_ids:
    frames = query.get_object_frames(obj_id)
    masks = query.load_masks_batch(obj_id, frames)
    # 將 masks 投影至 3D，並交由 AggregationPipeline 處理
```

聚合完成後，`proposal_data_level{L}.npz` 會成為 SigLIP assignment 模組（`my3dis.siglip_assignment`）與開放詞彙推論（`my3dis.inference`）的共同輸入。
因此，Aggregation 模組的正確性直接影響後續特徵指派與檢索結果的品質。

**注意事項（2025-11-19更新）**：
- Inference 模組（特別是 hierarchical 策略）已支持動態層級檢測
- 無需手動配置層級對應關係，系統會自動從可用的 proposal 文件中檢測
- 確保 aggregation 輸出的層級與 SAM2 追蹤輸出一致（通常為 L1/L3/L5）

