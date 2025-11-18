# SigLIP Feature Assignment 模組說明

本模組實作 3D 投影管線的第二階段：  
為 3D proposals 指派來自 SigLIP 的視覺／語意特徵，並可選擇性地進行開放詞彙分類，以支援後續的文字查詢與相似度檢索。

---

## 整體概觀

**輸入**：Aggregation 階段輸出的 3D proposals（`proposal_data_level{2,4,6}.npz`）  
**輸出**：加入 SigLIP 特徵與選配 labels 的 proposals（`siglip_features_level{2,4,6}.npz`）

### 管線流程（概念）

```text
proposal_data_level{L}.npz（來自 aggregation）
    ↓
對每個 proposal：
  - 根據 family tree / tracking metadata 找出最佳 K 張視角（frames）
  - 在每張圖上抽出多尺度的 2D crops
  - 將 crops 丟入 SigLIP 視覺編碼器
  - 在多幀、多尺度特徵之間做加權聚合
    ↓
（選配）利用文字查詢向量進行開放詞彙分類
    ↓
輸出含特徵／（選配）標籤的 proposals（NPZ）
```

---

## 目錄架構

```text
siglip_assignment/
├── __init__.py                 # 對外匯出 SigLIPAssignmentPipeline
├── siglip_pipeline.py          # 主要管線 orchestrator
├── feature_extraction.py       # SigLIP 視覺與文字特徵抽取
├── assignment_utils.py         # 開放詞彙推論與特徵聚合
└── README.md                   # 本文件
```

---

## 主要元件與流程

### 1. SigLIPAssignmentPipeline（`siglip_pipeline.py`）

此類別為整體指派流程的入口，負責連接 proposals、影像資料與 SigLIP 模型。

- `SigLIPAssignmentPipeline.__init__(proposal_dir, color_frames_dir, depth_maps_dir, model_name='google/siglip-so400m-patch14-384', batch_size=32, topk_frames=5, device='cuda:0')`
  - `proposal_dir`：aggregation 輸出所在目錄；
  - `color_frames_dir`：場景彩色影像所在目錄；
  - `depth_maps_dir`：對應深度圖所在目錄（若需要視角排序或遮蔽檢查）；
  - `model_name`：HuggingFace SigLIP 模型名稱；
  - `batch_size`：一次處理的 crop 數量；
  - `topk_frames`：每個 proposal 要取的最佳視角數目；
  - `device`：運算裝置（例如 `'cuda:0'` 或 `'cpu'`）。

- `run(output_dir, levels=None, object_labels=None) -> Dict[str, str]`
  - 若 `levels` 為 `None`，則自動偵測 `proposal_dir` 中具備 proposals 的層級；
  - 對每個層級：
    1. `_load_proposals(level)`：載入 `proposal_data_level{L}.npz`；
    2. `_extract_features_for_proposals(level_data)`：為所有 proposals 抽取 SigLIP 特徵；
    3. 若 `object_labels` 不為 `None`，則呼叫 `_assign_labels(features, object_labels)`，以文字查詢向量進行分類；
    4. `_export_results(level, payload)`：將結果寫成 `siglip_features_level{L}.npz`；
  - 統一由 `_export_metadata(outputs)` 寫出 `siglip_metadata.json`，紀錄模型名稱、batch size、topk_frames 與輸出路徑；
  - 回傳 `{str(level): npz_path}` 字典。

- `main()`  
  對應 CLI `python -m my3dis.run_siglip_assignment`，通常由 YAML workflow 的 `siglip_assignment` stage 或手動指令觸發：

```bash
PYTHONPATH=src python -m my3dis.run_siglip_assignment \
  --proposal-dir outputs/aggregation/scene_00065_00 \
  --color-frames-dir /path/to/scene/color \
  --depth-maps-dir /path/to/scene/depth \
  --output-dir outputs/siglip/scene_00065_00 \
  --levels 2 4 6 \
  --model google/siglip-so400m-patch14-384 \
  --batch-size 32 \
  --topk-frames 5 \
  --labels chair table sofa lamp \
  --device cuda:0
```

---

### 2. SigLIP 特徵抽取（`feature_extraction.py`）

#### 類別 `SigLIPFeatureExtractor`

- `__init__(model_name, device, cache_dir=None)`
  - 透過 HuggingFace 下載並載入對應 SigLIP 模型與 Processor；
  - 支援指定 GPU/CPU 以及模型快取位置。

- `extract_batch(frames, crops, mask_metadata)`
  - 主要用於批次處理多張影像與多個 proposal：
    - `frames`：影像檔路徑列表；
    - `crops`：每張影像中對應的 bounding boxes 或遮罩區域；
    - `mask_metadata`：額外資訊（例如 proposal id、level、view index 等）。
  - 函式流程：
    1. 透過 `_extract_from_frame` 逐張影像載入並裁切；
    2. 對每個 proposal 呼叫 `_extract_single_proposal` 產生多尺度 crops；
    3. 對所有 crops 做 batched 推論，得到嵌入向量；
    4. 將多尺度特徵平均或加權平均，產生最終 proposal 特徵。

- `get_text_features(labels)`
  - 將文字標籤（例如 `"chair"`, `"table"`）轉為 L2 正規化的 SigLIP 嵌入；
  - 供 `assignment_utils.assign_labels` 使用。

- `main()`  
  提供簡易 CLI，便於在非完整管線下測試特徵抽取性能與模型設定。

---

### 3. 指派工具與開放詞彙分類（`assignment_utils.py`）

- `aggregate_features(frame_features, weights=None)`
  - 將同一 proposal 在多幀、多視角的 SigLIP 特徵進行聚合；
  - 若提供 `weights`，則執行加權平均，否則為等權平均。

- `assign_labels(features, text_embeddings, temperature=300.0, allow_duplicates=True)`
  - **開放詞彙物件分類**：
    1. 以 cosine similarity 計算 proposal 特徵與文字嵌入的相似度；
    2. 透過溫度縮放（`temperature`）與 softmax 轉為概率；
    3. 根據配置決定是否允許不同 proposals 重複選取相同 label。
  - 輸出通常為 `(pred_classes, pred_scores)`。

- `assign_part_labels(object_features, part_features, temperature=300.0)`
  - 將物件與部件的特徵一起考量，適用於物件‑部件階層標籤；
  - 可看作 `assign_labels` 的擴展，將 object / part embeddings 共同放入相似度計算。

- `compute_mask_scores(proposal_masks, per_point_scores)`
  - 當混合使用點級 score（例如幾何／其他模型分數）時，將點級分數聚合為 proposal 級分數；
  - 典型作法為在每個 proposal mask 範圍內取平均或加權平均。

---

## 輸出格式

### `siglip_features_level{L}.npz`

每一層級的特徵檔大致包含：

```python
{
    # 由 aggregation 階段繼承
    'masks': np.ndarray,        # [N_points, N_proposals]  binary masks
    'obj_ids': np.ndarray,      # [N_proposals]            object IDs
    'levels': np.ndarray,       # [N_proposals]            level numbers
    'parent_ids': np.ndarray,   # [N_proposals]            parent IDs
    'volumes': np.ndarray,      # [N_proposals]            volumes (m^3)
    'centroids': np.ndarray,    # [N_proposals, 3]         centroids
    'bboxes': np.ndarray,       # [N_proposals, 6]         bboxes (min_xyz, max_xyz)
    'points': np.ndarray,       # [N_points, 3]            coordinates
    'colors': np.ndarray,       # [N_points, 3]            colors

    # 新增：SigLIP 特徵
    'features': np.ndarray,     # [N_proposals, D]         L2-normalized features

    # 若同時進行 label 指派，則額外包含：
    'pred_classes': np.ndarray, # [N_proposals]            predicted class indices
    'pred_scores': np.ndarray,  # [N_proposals]            confidence scores
}
```

### `siglip_metadata.json`

```json
{
  "proposal_dir": "aggregation_output/",
  "model": "google/siglip-so400m-patch14-384",
  "batch_size": 32,
  "topk_frames": 5,
  "outputs": {
    "2": "siglip_features_level2.npz",
    "4": "siglip_features_level4.npz",
    "6": "siglip_features_level6.npz"
  }
}
```

---

## 多尺度 cropping 策略（直觀說明）

對於每一個 proposal 在每一張選中的 frame 上，通常會進行下列操作：

```python
# 初始 crop：proposal 的 tight bounding box
crop_1 = image[y1:y2, x1:x2]

# 第二個尺度：將 bounding box 擴大 20%
expand = int(0.2 * max(y2 - y1, x2 - x1))
crop_2 = image[y1 - expand : y2 + expand,
               x1 - expand : x2 + expand]

# 第三個尺度：可能採用更大的擴張或其他啟發式
crop_3 = ...
```

在實際實作中，`SigLIPFeatureExtractor` 會在 RGB 空間中依照模型輸入大小進行 resize／normalize，並確保三個尺度的特徵在聚合前已對齊維度。

聚合的概念為：

```text
features_proposal = average( feature(crop_1), feature(crop_2), feature(crop_3) )
```

之後再在多幀之間進行第二層的加權平均。

---

## 使用建議與常見問題

### 1. 只需要特徵，不需要標籤

```python
from my3dis.siglip_assignment import SigLIPAssignmentPipeline

pipeline = SigLIPAssignmentPipeline(
    proposal_dir='aggregation_output/',
    color_frames_dir='color/',
    depth_maps_dir='depth/',
    model_name='google/siglip-so400m-patch14-384',
    batch_size=32,
    topk_frames=5,
)

results = pipeline.run(
    output_dir='siglip_output/',
    levels=[2, 4, 6],
    object_labels=None,  # 或省略此參數
)
```

### 2. 特徵 + 開放詞彙標籤

```python
results = pipeline.run(
    output_dir='siglip_output/',
    levels=[2, 4, 6],
    object_labels=['chair', 'table', 'sofa', 'lamp'],
)
```

此時輸出的 `siglip_features_level{L}.npz` 會包含 `pred_classes` 與 `pred_scores` 欄位，可直接供 Search3D 或其他評估工具使用。

### 3. 記憶體不足（Out of Memory）

可以嘗試：

- 降低 `batch_size`（例如改為 16 或 8）；
- 減少 `topk_frames`（例如由 5 降為 3）；
- 若在 GPU 上執行，確認其他程式是否佔用同一張 GPU 的記憶體。

### 4. 模型下載失敗

可事先在環境中下載模型：

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained('google/siglip-so400m-patch14-384')
processor = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')
```

下載成功後再執行 pipeline，便可從本地快取載入。

### 5. 預測效果不理想

- 適度增大 `topk_frames` 以增加視角數；
- 開啟或調整多尺度 cropping 策略；
- 使用更大的 SigLIP 變體（例如 `siglip-so400m`）；
- 在 `assignment_utils` 中調整溫度參數與 `allow_duplicates`。

---

## Module & Function Reference（補充）

### `siglip_pipeline.py`
- `SigLIPAssignmentPipeline.__init__`：建構整體 pipeline，設定 proposals 路徑、影像根目錄、SigLIP 模型名稱與推論裝置等。
- `run(output_dir, levels, object_labels)`：主流程函式，負責呼叫 `_load_proposals`、`_extract_features_for_proposals`、`_assign_labels`、`_export_results` 與 `_export_metadata`。
- `main()`：命令列入口，對應 `python -m my3dis.siglip_assignment.siglip_pipeline`，也可用於 smoke test。

### `feature_extraction.py`
- `SigLIPFeatureExtractor.__init__`：載入模型與 Processor，設定 device 與快取。
- `extract_batch(frames, crops, mask_metadata)`：對多張影像／多個 proposal 進行批次推論與特徵聚合。
- `_extract_single_proposal`／`_extract_from_frame`：封裝單一 proposal／影像的 crop 邏輯與前處理。
- `get_text_features(labels)`：將文字標籤轉成 SigLIP 嵌入，用於開放詞彙比對。
- `main()`：簡易 CLI，主要用於特徵抽取實驗與性能測試。

### `assignment_utils.py`
- `aggregate_features(frame_features, weights)`：在多幀特徵之間進行加權聚合。
- `assign_labels(features, text_embeddings, temperature, allow_duplicates)`：使用 cosine similarity 與溫度縮放對 proposals 進行開放詞彙分類。
- `assign_part_labels(object_features, part_features, temperature)`：同時考慮物件與部件特徵的標籤指派。
- `compute_mask_scores(proposal_masks, per_point_scores)`：將點級分數匯總為 proposal 級別分數，方便與 SigLIP 特徵一起使用。

---

## 後續整合

完成 SigLIP 指派後：

1. `inference` 模組可直接載入 `siglip_features_level{L}.npz`，利用文字查詢進行開放詞彙 3D 檢索；
2. 也可以將特徵匯出給其他研究專案，用於相似度搜尋或下游任務（例如 3D 場景理解、跨場景對齊等）。

相關模組：

- `../aggregation/` — 產生 3D proposals。
- `../family_tree_query.py` — 查詢 family tree 結構。
- `../inference/` — 完整開放詞彙推論管線。

