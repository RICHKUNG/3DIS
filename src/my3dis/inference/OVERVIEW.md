# Inference 模組總覽（開放詞彙 3D 實例檢索）

## 目的與角色

`my3dis.inference` 實作針對 MultiScan 資料集的**開放詞彙 3D 實例分割與檢索**，主要服務 Search3D 等基準測試。  
該模組以「先有 3D proposals + SigLIP 特徵」為前提，負責：

- 將文字或視覺查詢轉換為嵌入向量；
- 在多層級（物件／部件，L2/L4/L6）提案之間檢索候選；
- 融合幾何約束（包含度、尺度、中心距離等）與多實例 NMS；
- 最終輸出符合 Search3D 格式的預測結果。

## 目錄架構

```text
my3dis/inference/
├── __init__.py              # 對外匯出的主要介面
├── config.py                # 組態 dataclass（YAML ↔ 物件）
├── feature_extraction.py    # SigLIP 特徵載入與抽取（inference 端）
├── retrieval.py             # 多層級候選檢索
├── pairing.py               # 物件／部件配對與 family tree 互動
├── nms.py                   # 多實例 NMS 與抑制策略
├── scoring.py               # 分數聚合與溫度縮放
├── geometry.py              # 幾何約束（IoU、包含度、尺度）
├── formatter.py             # Search3D 輸出格式化與驗證
├── inference_pipeline.py    # 封裝整體推論流程的 Pipeline 類別
└── strategies/
    ├── __init__.py
    ├── base.py              # 策略基底類別
    ├── independent.py       # 物件／部件獨立檢索策略
    ├── hierarchical.py      # 利用 family tree 的階層式策略
    ├── exhaustive_pairing.py                    # 全組合配對策略
    └── exhaustive_pairing_with_combined_query.py # 加入 combined query 的暴力策略
```

## 模組相依關係（概念圖）

```text
retrieval.py
    └─> geometry.py        （IoU / 幾何相似度，用於 dedup 與候選分數）

pairing.py
    ├─> retrieval.py       （Proposal dataclass）
    ├─> geometry.py        （包含度／尺度檢查）
    └─> scoring.py         （分數聚合）

nms.py
    └─> geometry.py        （IoU、中心距離等）

strategies/
    ├─> retrieval.py
    ├─> pairing.py
    ├─> nms.py
    └─> scoring.py

inference_pipeline.py
    ├─> retrieval.py
    ├─> pairing.py
    ├─> nms.py
    ├─> scoring.py
    ├─> formatter.py
    └─> strategies/

formatter.py
    ├─> nms.py             （Prediction dataclass）
    └─> pairing.py         （ObjectPartPair dataclass）
```

---

## 核心資料結構

### `Proposal`

```python
@dataclass
class Proposal:
    level: str              # 'L2', 'L4', or 'L6'
    proposal_id: int        # index within level
    mask: np.ndarray        # binary mask over points (N_points,)
    feature: np.ndarray     # SigLIP embedding (D,)
    object_id: int          # from tracking / family tree
    score: float            # similarity score
```

代表一個「3D proposal + 對應特徵」，由 aggregation 與 SigLIP assignment 串接而來。

### `ObjectPartPair`

```python
@dataclass
class ObjectPartPair:
    object_proposal: Proposal
    part_proposal: Proposal
    object_score: float
    part_score: float
    geometric_score: float
    combined_score: float
    label_id: int           # joint semantic label
```

代表一組「物件 proposal + 部件 proposal」的配對關係，包含各自分數與幾何／綜合分數。

### `Prediction`

```python
@dataclass
class Prediction:
    mask: np.ndarray        # binary mask
    score: float            # confidence
    label_id: int           # semantic label
    metadata: dict          # additional info
```

代表最後輸出的單一預測，`formatter` 會將一組 `Prediction` 轉換成 Search3D 所需的矩陣。

---

## 策略執行流程

### Independent Strategy（物件／部件獨立檢索）

```text
1. 物件候選檢索
   ├─ 從 L2 檢索 top-K
   ├─ 從 L4 檢索 top-K
   └─ 以 IoU > 0.95 去重
       └─ 得到 Object Pool（M 個候選）

2. 部件候選檢索
   ├─ 從 L4 檢索 top-K
   ├─ 從 L6 檢索 top-K
   └─ 去重
       └─ 得到 Part Pool（N 個候選）

3. 幾何配對
   對於 Object Pool × Part Pool 中每組 (object, part)：
       ├─ 檢查包含度比率 > 閾值
       ├─ 檢查尺度比率是否在合理範圍内
       ├─ 計算幾何分數
       └─ 聚合 object / part / 幾何分數
           └─ 得到有效配對集合（P 組）

4. 多實例 NMS
   依分數排序：
       對每個配對：
           ├─ 計算與已保留配對的 IoU
           ├─ 若 IoU > 閾值：
           │   ├─ 檢查中心距離
           │   └─ 若距離太近則抑制（視為重複實例）
           └─ 否則保留
               └─ 得到最終預測（Q 筆）
```

### Hierarchical Strategy（階層式 family‑aware 策略）

```text
1. 粗略定位（L2）
   └─ 從 L2 檢索 top-K proposal
       └─ Coarse Candidates

2. 多解析度 refinement
   對於每一個 L2 候選：
       ├─ 保留 L2 proposal
       ├─ 從 family tree 取得其 L4 子節點
       ├─ 以 object 查詢重新評分 L4 候選
       └─ 若分數高於閾值或高於 parent + margin：
           └─ 加入物件候選集（L2 + L4 混合）

3. 區域內部件搜尋
   對於每一個物件候選：
       ├─ 從家族樹搜尋其子孫節點（L4 / L6）
       ├─ 以 part 查詢評分
       ├─ 依閾值篩選
       └─ 組成物件／部件配對
           └─ Candidate Pairs

4. 跨路徑 NMS
   └─ 運用多實例 NMS，避免家族樹不同分支產生高度重疊的重複預測
       └─ Final Predictions
```

---

## 組態階層（`config.py`）

主要透過 dataclass 將 YAML 解析後的設定封裝，並提供序列化／反序列化與策略參數拆解。

```text
InferenceConfig
├── strategy: str                         # 使用的策略名稱，如 'independent'、'hierarchical'
├── retrieval: RetrievalConfig
│   ├── temperature                       # 溫度縮放（相似度 → 概率）
│   ├── min_similarity                    # 最低相似度門檻
│   └── top_k_per_level                   # 各層級的 top-K
├── pairing: PairingConfig
│   ├── containment_threshold             # 物件/部件的包含度門檻
│   └── scale_range                       # 尺度比率允許範圍
├── nms: NMSConfig
│   ├── iou_threshold                     # NMS IoU 閾值
│   └── centroid_distance_ratio           # 中心距離比率閾值
├── scoring: ScoringConfig
│   ├── method                            # 分數聚合方法
│   └── weights                           # 各分數權重
├── formatter: FormatterConfig            # 輸出格式相關參數
├── independent: IndependentStrategyConfig
└── hierarchical: HierarchicalStrategyConfig
```

---

## Search3D 輸出格式

### 結構

```python
{
    'pred_classes': np.ndarray,  # Shape: (M,), dtype: int32
    'pred_scores': np.ndarray,   # Shape: (M,), dtype: float
    'pred_masks': np.ndarray     # Shape: (P, M), dtype: uint8
}
```

其中：

- `P`：場景內點的數量（點雲大小）；
- `M`：輸出預測的數量；
- `pred_masks[:, j]`：第 `j` 個預測之二值遮罩（對所有點）。

`formatter.Search3DFormatter` 負責將 `Prediction` 集合轉成上述格式，並提供驗證工具。

---

## 模組與函式參考

### `config.py`
- Dataclass：
  - `RetrievalConfig`, `PairingConfig`, `NMSConfig`, `ScoringConfig`,  
    `FormatterConfig`, `IndependentStrategyConfig`, `HierarchicalStrategyConfig`, `ExhaustivePairingConfig`：
    - 各自對應檢索、配對、NMS、分數、輸出與策略之超參數；
    - 欄位與 YAML key 一一對應，方便由設定檔直接構造實例。
  - `InferenceConfig`：
    - `from_yaml(path_or_dict)`：從 YAML 檔案或 dict 建構完整推論設定；
    - `to_yaml(path)`／`to_dict()`：將目前設定寫回 YAML 或 dict，方便紀錄與重現實驗；
    - `get_strategy_kwargs()`：為指定策略建立 kwargs（例如給 `IndependentStrategy` 或 `HierarchicalStrategy`）。

### `feature_extraction.py`
- `SigLIPFeatureExtractor`：
  - `__init__(model_name, device)`：載入 HuggingFace SigLIP/CLIP 模型與 Processor，支援 GPU／CPU 選擇。模型載入失敗時會直接拋出 RuntimeError。
  - `extract_text_features(text_queries, batch_size)`：將文字查詢轉為 L2 正規化嵌入，用於 cosine similarity。
  - `extract_proposal_features(proposals_by_level, exp_dir, scene_path, max_frames_per_object, batch_size)`：
    - 由追蹤輸出載入 proposal 的 RGB 影像裁切；
    - 尋找各 level 目錄（`_find_level_dir`）；
    - 透過 `_extract_single_proposal_feature` 提取並聚合特徵（加權時序+尺寸）；
    - 失敗時直接拋出 RuntimeError，不使用隨機特徵。

### `retrieval.py`
- `compute_cosine_similarity(query, features)`：
  - 計算查詢向量與候選特徵矩陣之間的 cosine similarity，為檢索與重新排序的基礎。
- Dataclass：
  - `Proposal`／`RetrievalResult`：分別代表單個 proposal 與一系列檢索結果（包含分數與排序）。
- `MultiLevelRetriever`：
  - `__init__(proposals_by_level, config)`：接收預先載入好的 proposals 與檢索設定；
  - `retrieve_single_level(level, query, top_k)`：
    - 在單一層級執行檢索；
    - 套用 `min_similarity` 與 IoU-based dedup（透過 `_deduplicate_proposals`）；
  - `retrieve_multi_level(query)`：
    - 同時對多個層級進行檢索；
    - 整合結果並保持層級資訊；
  - `retrieve_for_object_part(object_query, part_query)`：
    - 為物件／部件查詢分別取得候選集合，供 pairing 模組使用。

### `pairing.py`
- 工具：
  - `parse_level_to_prefix(level)`：將 level 數字（2/4/6）轉為 `'L{level}'` 字串，對齊 family tree 的 key。
- 資料結構與類別：
  - `ObjectPartPair`：如前所述，表示一次物件／部件配對結果。
  - `FamilyTree`：
    - `__init__(family_tree_json)`：以 `family_tree.json` 建立內部父子關係圖；
    - `get_children(obj_id)`／`get_descendants(obj_id)`：返回子結點與整個子孫集合；
    - `is_ancestor(ancestor_id, descendant_id)`：檢查兩者是否為祖先／後代關係。
  - `ObjectPartPairer`：
    - `__init__(config, geometry_cfg)`：綁定 pairing 與幾何相關設定；
    - `pair_all(object_pool, part_pool, thresholds)`：
      - 檢查 `_is_feasible_pair`（包含度、尺度與其他限制）；
      - 產生 `ObjectPartPair` 列表並計算綜合分數。
    - `pair_with_tree_guidance(object_pool, part_pool, family_tree, thresholds)`：
      - 額外利用 `FamilyTree`，僅在祖先／後代路徑上搜尋部件候選。

### `geometry.py`
- 基本幾何運算：
  - `compute_iou(mask_a, mask_b)`：在點雲遮罩或投影平面上計算 IoU；
  - `compute_containment_ratio(object_mask, part_mask)`：計算部件在物件遮罩中的覆蓋比例；
  - `compute_centroid(mask)`／`compute_centroid_distance(mask_a, mask_b)`：計算實例中心及其距離；
  - `compute_scale_ratio(object_mask, part_mask)`：計算兩個遮罩的尺度關係。
- 綜合檢查：
  - `check_geometric_validity(object_mask, part_mask, thresholds)`：
    - 同時檢查包含度、尺度與距離等條件，作為 pairing 與 NMS 的前置篩選；
  - `compute_geometric_score(...)`：
    - 將上述幾何量轉為單一分數，用於 `ObjectPartPair.combined_score`。

### `nms.py`
- Dataclass：
  - `Prediction`：前述代表單一預測（mask, score, label, metadata）。
- 函式：
  - `standard_nms(predictions, iou_threshold)`：
    - 傳統 class-aware NMS，依 label 分組、IoU 抑制高重疊候選。
  - `multi_instance_nms(pairs, iou_threshold, centroid_ratio)`：
    - 適用於物件／部件配對結果；
    - 在 IoU 過高情況下，會檢查中心距離，保留空間上明顯分離的實例。
  - `class_agnostic_nms(predictions, iou_threshold)`：
    - 不區分類別，僅依 IoU 抑制（常用於某些分析或 ablation）。
  - `nms_with_instance_center_check(predictions, iou_threshold, centroid_ratio)`：
    - 將 multi-instance 中心距離檢查概念套用到一般 `Prediction` 集合。

### `scoring.py`
- 定義分數聚合策略：
  - `ScoreAggregationMethod`：枚舉可用的方法（例如加權和、乘積等），便於 log 與設定。
  - `ScoreAggregator`：
    - `aggregate(object_score, part_score, geometric_score)`：將三類分數聚合為單一 scalar；
    - `aggregate_with_hierarchy_bonus(object_score, part_score, geometric_score, is_same_family)`：
      - 若物件與部件位於同一 family tree 路徑上，則適度提高分數。
  - `AdaptiveScoreAggregator`：
    - `aggregate(...)`：在多種聚合策略中自動選擇最佳者，或做折衷加權。
- 輔助函式：
  - `apply_temperature_scaling(logits, temperature)`：將 raw score 透過 softmax＋溫度縮放為概率分佈；
  - `compute_confidence_distribution(scores)`：生成可視化或後處理用的信心分佈資訊。

### `formatter.py`
- `Search3DFormatter`：
  - `__init__(config)`：綁定 formatter 設定（如是否 class‑agnostic、是否輸出額外 metadata）。
  - `format_predictions(predictions)`：
    - 將 `Prediction` 集合轉換成 Search3D 格式的 `pred_classes`／`pred_scores`／`pred_masks`。
  - `format_pairs(pairs)`：
    - 將 `ObjectPartPair` 集合轉換成較豐富的結構，保留物件／部件關係與分數構成。
  - `format_multi_scene(scene_to_predictions)`：
    - 將多場景預測整合為單一 payload，用於批次評估。
  - `save_predictions(path, payload)`／`load_predictions(path)`：
    - 寫入／讀取 JSON + npz 組合，方便實驗記錄與重現。
- `validate_prediction_format(payload)`：
  - 檢查欄位存在與 shape/dtype 是否符合 Search3D 要求，於提交前自檢。

### `inference_pipeline.py`
- `InferencePipeline`：
  - `__init__(config, tracking_root, family_tree_path, feature_cache_dir)`：
    - 建立 feature extractor、retriever、pairer、策略與 formatter；
    - 若需 family tree，則透過 `_load_family_tree` 快取讀取。
  - `_load_family_tree(path)`：
    - 載入 `family_tree.json` 並建立 `FamilyTree` 物件，供階層式策略使用。
  - `_create_strategy(name)`：
    - 根據 `InferenceConfig.strategy` 建立對應策略實例（`IndependentStrategy`、`HierarchicalStrategy` 等），並傳入適當 kwargs。
  - `infer_single_query(object_text, part_text)`：
    - 針對單一查詢（物件／部件描述）執行完整 pipeline（檢索 → 配對 → NMS → 格式化）。
  - `infer_batch_queries(queries)`：
    - 針對多個查詢批次處理，重用已載入的特徵與 proposals。
  - `format_predictions(predictions)`：
    - 呼叫 formatter 將內部 `Prediction` 轉成 Search3D payload。
  - `save_predictions(path, payload)`：
    - 將預測結果寫入檔案（常與 config snapshot 一起存放）。
- `load_proposals_from_tracking_output(run_dir, levels)`：
  - 由追蹤／聚合輸出載入 proposals，為離線推論或分析提供便捷介面。

### `strategies/`（推論策略族）
- `base.py` — `InferenceStrategy`：
  - `__init__(retriever, pairer, nms_module, formatter, config)`：
    - 將檢索、配對、NMS 與 formatter 串起來；
  - `infer(object_query, part_query)`（抽象）：
    - 由子類別實作整體流程；
  - `apply_nms_to_pairs(pairs)`：
    - 將配對結果丟給適當的 NMS 函式；
  - `pairs_to_predictions(pairs)`：
    - 將 `ObjectPartPair` 轉成 `Prediction` 集合。
- `combined_query_mixin.py` — `CombinedQueryMixin`：
  - `setup_combined_query(config)`／`create_combined_query(object_text, part_text)`：
    - 將物件與部件文字提示合併成單一查詢嵌入；
  - `extract_combined_feature(...)`／`update_proposals_with_combined_feature(...)`：
    - 為 proposals 附加 combined feature，支援策略內部反覆使用；
  - `retrieve_with_combined_query(...)`：
    - 使用 combined feature 進行檢索；
  - `check_combined_query_enabled(config)`：
    - 檢查是否在 config 中啟用了 combined query 模式。
- `independent.py` — `IndependentStrategy`：
  - `infer(...)`：
    - 實作本文件前述之「物件／部件獨立檢索策略」；
  - `_retrieve_with_backoff(...)`／`_build_threshold_schedule(...)`／`_has_level_candidates(...)`：
    - 針對各層級建立階梯式閾值策略（若 top‑K 不足則放寬條件），避免某層級完全沒有候選；
  - `infer_single_level(...)`：僅在單一層級上推論；
  - `infer_with_text_queries(...)`：針對文字查詢執行完整 pipeline。
- `hierarchical.py` — `HierarchicalStrategy`：
  - `infer(...)`：
    - 實作本文件前述之「階層式 family‑aware 策略」；
  - `_coarse_localization`／`_refine_objects`／`_search_parts_for_objects`／`_get_part_candidates_for_object`：
    - 各階段負責粗略定位、物件 refinement、部件搜尋與候選收斂；
  - `infer_with_text_queries(...)` 及 `_..._with_combined_feature` 系列：
    - 提供啟用 combined query 後的變體實作。
- `exhaustive_pairing.py` — `ExhaustivePairingStrategy`：
  - 以暴力方式枚舉所有層級配對組合，主要用於研究與診斷：
    - `_generate_level_pairs(...)`：建立所有 level 配對；
    - `_retrieve_with_backoff`／`_build_threshold_schedule`／`_has_level_candidates`：類似 `IndependentStrategy`；
    - `infer(...)`：在完整層級組合上執行配對與 NMS；
    - `get_level_pair_statistics()`：回傳各層級配對的候選數與分數分布統計。
- `exhaustive_pairing_with_combined_query.py`：
  - `ExhaustivePairingWithCombinedQuery`：
    - 在暴力策略上加入 combined query 支援；
    - `infer_with_text_queries(...)`：針對文字查詢執行完整暴力配對；
    - `get_level_pair_statistics_with_text(...)`：提供文字查詢情境下的層級配對統計。

---

若需要將本模組與整體 pipeline 聯繫起來，可搭配閱讀：

- 3D proposal 產生：`src/my3dis/aggregation/README.md`
- SigLIP 特徵指派：`src/my3dis/siglip_assignment/README.md`
- 全體系統流程：`src/my3dis/OVERVIEW.md`

