# My3DIS mAP 實驗整合總結

## 1. 資料來源
- `eval/COMPLETION_SUMMARY.md`、`eval/search3d/FORMAT_VALIDATION_REPORT_scene_00005_00.md`
- `MAPerformance_DIAGNOSIS.md`、`AUTO_LEVEL_DETECTION_FIX_2025_11_16.md`
- `OPTIMIZATION_PLAN.md`、`OPTIMIZATION_RESULTS.md`
- `CONFIG_OPTIMIZATION_SUMMARY.md`、`STRATEGY_FAILURE_ANALYSIS.md`
- `FINAL_FIX_SUMMARY.md`、`EMERGENCY_FIX_APPLIED.md`、`HIERARCHICAL_RESTORE_FIX.md`
- `CRITICAL_DIAGNOSIS_NO_BASELINE.md`、`PLAN_A_EXPERIMENT_TRACKER.md`
- `PHASE1_EXPERIMENT_LOG.md`、`PHASE1_EXPERIMENT_STATUS.md`、`PHASE1_FAILURE_ANALYSIS.md`
- `COMPREHENSIVE_FIX_PLAN.md`、`COMPLETE_DIAGNOSIS_AND_SOLUTION_2025_11_17.md`
- `DIAGNOSTIC_RESULTS_2025_11_17.md`、`logs/eval/PROPOSAL_GT_OVERLAP_scene_00093_01.md`
- `logs/eval/EXPERIMENT_2025_11_17_3Dsiglip_independent.md`、`logs/eval/SIGLIP_RETRIEVAL_PLAN_2025_11_17.md`
- `SIGLIP_FEATURE_FIX_SUMMARY.md`、`CODE_LEVEL_OPTIMIZATION_ANALYSIS.md`
- 其他內含 mAP 討論的筆記（例如 `MULTI_SCENE_PIPELINE_GUIDE.md`、`COMPREHENSIVE_FIX_PLAN.md` 等）

## 2. 關鍵里程碑與量測
| 日期 | 策略 / 配置 | 報告 | 觀察到的 mAP / 指標 | 核心結論 |
|------|-------------|------|----------------------|-----------|
| 2025-11-07 | Quick eval + Search3D | `eval/COMPLETION_SUMMARY.md` | mAP=0（評估失敗） | Pipeline 可完整跑通，但因 2D mask 與 3D GT 長度不符而無法計算 mAP，需要 2D→3D 投影。 |
| 2025-11-13 | search3d 格式檢查 | `eval/search3d/FORMAT_VALIDATION_REPORT_scene_00005_00.md` | mAP≈0 | Object 層級雖格式正確，但僅 4/19 GT 類別被命中；Object-Part 層級因預測使用語義 composite ID 而和實例 ID 完全不重疊。 |
| 2025-11-16 | 聚合 + 推理優化 | `OPTIMIZATION_RESULTS.md` | Hierarchical mAP=0.006、AP50=0.050、Max IoU=0.5385 | 降低 aggregation IoU/area 閾值與提高 top-k 後，首次產生 AP>0 的結果，但僅 1 個匹配達 IoU≥0.5。 |
| 2025-11-16 | 自動層級偵測 | `AUTO_LEVEL_DETECTION_FIX_2025_11_16.md` | 產生預測但 mAP=0 | 修復 L1/L3/L5 配置錯誤後恢復 79-711 筆預測，證實 mAP=0 已不再是「無輸出」造成。 |
| 2025-11-16 | Emergency Fix | `EMERGENCY_FIX_APPLIED.md` | 預期 Hierarchical mAP=0.008-0.020（尚未驗證） | 移除幾何約束 + Soft NMS 以提升候選，但缺乏實際評估證據。 |
| 2025-11-16 | 評估路徑調查 | `FINAL_FIX_SUMMARY.md` | 實際 Hierarchical mAP=0.006；22:14 run 顯示 0.000 | mAP=0 來自評估指向錯誤檔案；修正路徑即可還原 0.006。 |
| 2025-11-16 | 策略比較 | `STRATEGY_FAILURE_ANALYSIS.md` | Independent/Exhaustive mAP=0、Hierarchical=0.006 | Hierarchical 借助 family tree 保留 110 預測，其餘策略因 top-k/幾何限制僅剩 13-22 預測。 |
| 2025-11-17 | 基線追蹤 | `CRITICAL_DIAGNOSIS_NO_BASELINE.md`, `PLAN_A_EXPERIMENT_TRACKER.md` | 所有重製測試 mAP=0 | 尚未能重現 18:00 的 0.006；需要逐項測試（A0-A3）來定位。 |
| 2025-11-17 | Phase 1 失敗 | `PHASE1_FAILURE_ANALYSIS.md` | Hierarchical mAP 從 0.006 → 0.0 | 同時關閉 softmax + 放寬幾何造成閾值錯配與噪音放大，430 預測仍全數 IoU<0.25。 |
| 2025-11-17 | SigLIP 診斷 | `DIAGNOSTIC_RESULTS_2025_11_17.md` | Oracle AP25=0.6296、實際 mAP=0 | SAM2→3D proposals 幾何上限良好，但 SigLIP 特徵標準差僅 0.007，語義分數幾乎不分辨。 |
| 2025-11-17 | Proposal 覆蓋分析 | `logs/eval/PROPOSAL_GT_OVERLAP_scene_00093_01.md` | IoU≥0.25 覆蓋 63% GT | L4/L6 仍能提供 IoU>0.7 的候選，證實語義排序及策略設定才是瓶頸。 |
| 2025-11-17 | Independent 回歸測試 | `logs/eval/EXPERIMENT_2025_11_17_3Dsiglip_independent.md` | 多個配置皆 mAP=0 | 無論 combined query、softmax、幾何權重怎麼調整，獨立策略最多 7 筆預測且全部 IoU<0.25。 |

## 3. 系統性問題歸納
### 3.1 評估與資料一致性
- 早期（11/07-11/13）的 mAP=0 主因是 Search3D 評估需要 3D 點雲實例 ID，而 pipeline 只輸出 2D masks 或語義 composite ID，導致評估中斷或零交集（`eval/COMPLETION_SUMMARY.md`、`eval/search3d/FORMAT_VALIDATION_REPORT_scene_00005_00.md`）。
- `FINAL_FIX_SUMMARY.md` 也顯示若評估讀錯策略子資料夾，會把有效預測當成 0。後續流程必須在單/多策略模式下都明確提供推理子路徑。
- GT 覆蓋率僅 34.7%（`MAPerformance_DIAGNOSIS.md`），且含有非法 ID（如 11001），在評估時容易造成查詢缺失與潛在的 false negative。

### 3.2 Pipeline 配置與策略差異
- 層級設定錯誤會直接讓所有策略零輸出，所幸自動偵測已在 11/16 修復（`AUTO_LEVEL_DETECTION_FIX_2025_11_16.md`）。
- 相同的 Emergency 參數對 Independent 有利、對 Hierarchical 卻有害；階層策略需要嚴格幾何約束與 hard NMS，而獨立策略反而需要完全放鬆（`HIERARCHICAL_RESTORE_FIX.md`、`STRATEGY_FAILURE_ANALYSIS.md`）。
- `PHASE1_FAILURE_ANALYSIS.md` 和 `CRITICAL_DIAGNOSIS_NO_BASELINE.md` 強調：關閉 softmax 後如果不把所有閾值重新定義在 [-1,1] 之間，會讓噪音全數闖關，mAP 直接歸零。
- 目前 `CONFIG_OPTIMIZATION_SUMMARY.md` 把 Hierarchical 設為唯一策略，確保至少能複現 0.006；然而 `PLAN_A_EXPERIMENT_TRACKER.md` 仍需要證實 baseline 真的可重現，再逐項加入 scale/softmax 改動。

### 3.3 特徵與檢索瓶頸
- `SIGLIP_FEATURE_FIX_SUMMARY.md` 指出 11/16 以前部分環境會回落到亂數向量，已透過 `3Dsiglip` 環境與 batched extractor 修復，但 `DIAGNOSTIC_RESULTS_2025_11_17.md` 顯示即便是正確特徵，對「door of cabinet」這類組合查詢的標準差仍 <0.01，語義幾乎無法區分。
- `CODE_LEVEL_OPTIMIZATION_ANALYSIS.md` 建議加入品質權重、adaptive bbox、mask-aware 特徵，以避免平均池化稀釋訊號；`logs/eval/SIGLIP_RETRIEVAL_PLAN_2025_11_17.md` 則展示了多輪 SigLIP 參數探索仍無法找出 IoU≥0.25 的匹配。

### 3.4 Proposal 幾何上限
- `logs/eval/PROPOSAL_GT_OVERLAP_scene_00093_01.md` 與 `DIAGNOSTIC_RESULTS_2025_11_17.md` 的 Oracle mAP/IoU 分析證明 L4/L6 其實提供了 0.7 以上 IoU 的候選，63% GT 在 ≥0.25；因此問題不在 3D 聚合，而在下游配對與排序。
- `OPTIMIZATION_PLAN.md`、`COMPREHENSIVE_FIX_PLAN.md` 建議放寬 pairing containment、降低 aggregation IoU、增加 top-k，以提高 IoU 覆蓋；`OPTIMIZATION_RESULTS.md` 顯示這些改動至少能讓一個實例達到 IoU 0.5385。

## 4. 策略別觀察
- **Hierarchical**：唯一多次達成非零 mAP 的策略（0.006），依賴 family tree 和嚴格幾何。需要針對 coarse/object/part top-k 與 threshold 進行精準調參；若一次調太多（Phase 1）就會崩潰。
- **Independent**：需要完全放鬆幾何限制才能產生預測，但 SigLIP 排名仍找不到高 IoU proposal；`EMERGENCY_FIX_APPLIED.md` 估計可達 0.003-0.015，但最新 11/17 測試尚未複現。
- **Exhaustive Pairing**：受限於 level 組合與高閾值，目前所有實驗仍為 0。需要限制有效的 level pair 並回傳所有組合以供最終 NMS 聚合（`STRATEGY_FAILURE_ANALYSIS.md`）。
- **Multi-scene / Pipeline 工具**：`MULTI_SCENE_PIPELINE_GUIDE.md` 證明多場景報表與 aggregated metrics 已實作完成，可在後續量產測試時提供自動報告，但尚未有實際 Search3D 資料的成功案例。

## 5. 目前進展與下一步建議
1. **重現 0.006 Baseline**：依 `PLAN_A_EXPERIMENT_TRACKER.md` 先完成 A0（原 18:00 配置）驗證，再逐一測試 scale 與 softmax 設定，避免 Phase 1 再次一次改動多項參數。
2. **SigLIP 特徵加強**：按照 `CODE_LEVEL_OPTIMIZATION_ANALYSIS.md` 的建議加入視圖品質權重、adaptive bbox 及 mask-aware pooling，並考慮 `SIGLIP_FEATURE_FIX_SUMMARY.md` 所列的替代模型（OpenCLIP 等）以增加特徵辨識度。
3. **策略特定參數覆蓋**：落實 `HIERARCHICAL_RESTORE_FIX.md` 內建議，為不同策略提供獨立的 pairing/NMS 設定，以便同時調優 Hierarchical 與 Independent。
4. **評估健全性**：持續檢查 GT ID（`MAPerformance_DIAGNOSIS.md`）、路徑與格式（`FINAL_FIX_SUMMARY.md`、`eval/search3d/FORMAT_VALIDATION_REPORT_scene_00005_00.md`），確保新的非零 mAP 不是因為評估器讀錯資料。
5. **診斷自動化**：善用 `logs/eval/PROPOSAL_GT_OVERLAP_scene_00093_01.md` 的 IoU 腳本來監控 proposal 上限，並在 `PHASE1_EXPERIMENT_STATUS.md` 所示的多場景流程中接入 aggregated_metrics，自動檢查是否有任何場景突破 0。

> 本文件將隨後續實驗更新，維持所有 mAP 記錄、成敗原因與推薦動作的單一總覽。
