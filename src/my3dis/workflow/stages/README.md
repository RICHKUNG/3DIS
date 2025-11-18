# `workflow/stages/` Stage Runners

`StageRunner` 衍生類別定義了 YAML 工作流程中的每一個 stage。所有類別都提供 `name`、`should_run()` 與 `execute()`，此外各 stage 會實作額外方法處理複雜邏輯。

## SSAMStageRunner (`ssam_stage.py`)
- `name` →  returns ``"ssam"``，用於 summary 與日誌鍵值。
- `should_run()` 讀取 stage 設定中 `enabled`，預設為 True。
- `execute()` 根據 `enabled` 決定要呼叫 `_run_candidate_generation()` 還是 `_handle_reuse_existing()`。
- `_handle_reuse_existing()` 嘗試從 `experiment.run_dir` 或輸出根目錄尋找現成的 `manifest.json`，成功後會直接更新 `SceneContext._run_dir/_manifest`。
- `_run_candidate_generation()` 透過 `SSAMStageConfig` 驗證 YAML，呼叫 `generate_candidates.run_generation()`，並利用 `StageRecorder` 記錄執行時間、GPU 與參數摘要。

## FilterStageRunner (`filter_stage.py`)
- 負責在 SSAM 後重新套用面積/穩定度限制。
- `should_run()` 會在 YAML 設定未指定或 `enabled=True` 時返回 True。
- `execute()` 匯入 `filter_candidates.run_filtering()`，讀取 `filtered` 子節點，完成後更新 workflow summary，包括各 level 的 kept/drop 統計。

## TrackerStageRunner (`tracker_stage.py`)
- `name` 為 `tracker`，對應 SAM2 追蹤。
- `should_run()` 預設 True，若 YAML 將 stage 關閉時才會略過。
- `execute()` 準備 `tracking` 模組需要的 context（資料路徑、SSAM manifest、GPU 指定），然後呼叫 `track_from_candidates.run_tracking()`，最後把 `video_segments`/`object_segments` 的路徑與長尾提示設定寫入 summary。

## FamilyTreeStageRunner (`family_tree_stage.py`)
- 依據 `family_tree` stage 設定決定是否產生 `family_tree.json`。
- `execute()` 呼叫 `family_tree_builder.build_family_tree()`，成功後將輸出路徑記錄在 summary。一般在 tracking 完成且 `relations/` 可用時才會執行。

## FamilyVizStageRunner (`family_viz_stage.py`)
- 用于產生家族樹可視化結果（例如特定 family 的 JSON/PNG）。
- `should_run()` 檢查 YAML 中是否提供目標 object/family 清單。
- `execute()` 會載入 `family_tree_query.FamilyTreeQuery`，產生指定 family 的子圖與附檔，供 QA 或 demo 使用。

## ReportStageRunner (`report_stage.py`)
- 控制 `generate_report.build_report()` 的執行，通常於所有 stage 完成時產出 Markdown 報告。
- `execute()` 支援覆寫輸出檔名與圖檔寬度，並把報告路徑存進 summary。

## __init__.py
- Re-exports stage類別，方便 `workflow.scene_workflow` 以 `from .stages import SSAMStageRunner, ...` 方式載入。EOF
