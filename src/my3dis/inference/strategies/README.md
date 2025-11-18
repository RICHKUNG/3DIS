# `inference/strategies/`

| 檔案 | 重點類別 / 函數 | 說明 |
| --- | --- | --- |
| `base.py` | `InferenceStrategy.__init__`, `infer`, `apply_nms_to_pairs`, `pairs_to_predictions` | 抽象策略基底。子類別共用檢索器、配對器、NMS 與 formatter，並經由 `infer()` 實作整體流程。NMS/轉換輔助函數確保所有策略輸出一致的 `Prediction` 物件。 |
| `combined_query_mixin.py` | `CombinedQueryMixin.setup_combined_query`, `create_combined_query`, `extract_combined_feature`, `update_proposals_with_combined_feature`, `retrieve_with_combined_query`, `check_combined_query_enabled` | 混合查詢功能，允許物件/部件文字提示合併為單一向量並套用於 `IndependentStrategy` 或 `ExhaustivePairingStrategy`。 |
| `independent.py` | `IndependentStrategy.__init__`, `infer`, `_retrieve_with_backoff`, `_build_threshold_schedule`, `_has_level_candidates`, `infer_single_level`, `infer_with_text_queries` | 兩段式策略：獨立檢索物件/部件、進行幾何配對、再以 multi-instance NMS 出結果；備援檢索與閾值排程確保不同層級都有候選。 |
| `hierarchical.py` | `HierarchicalStrategy.__init__`, `infer`, `_coarse_localization`, `_refine_objects`, `_search_parts_for_objects`, `_get_part_candidates_for_object`, 以及對應的 `_..._with_combined_feature` | 依據家族樹自粗到細搜尋，並支援混合查詢管線。每個 `_` 函數皆與 `family_tree_query.FamilyTreeQuery` 互動以取得親子節點。 |
| `exhaustive_pairing.py` | `ExhaustivePairingStrategy.__init__`, `_generate_level_pairs`, `infer`, `_retrieve_with_backoff`, `_build_threshold_schedule`, `_has_level_candidates`, `get_level_pair_statistics` | 暴力枚舉所有層級組合的策略，主要用於研究/診斷，並可輸出層級組合的統計資料。 |
| `exhaustive_pairing_with_combined_query.py` | `ExhaustivePairingWithCombinedQuery.__init__`, `infer_with_text_queries`, `get_level_pair_statistics_with_text` | 在暴力策略上加上混合查詢功能，確保文字+視覺特徵同步更新。 |
| `__init__.py` | 重新匯出上述策略供 `InferencePipeline` 直接 import。 |
