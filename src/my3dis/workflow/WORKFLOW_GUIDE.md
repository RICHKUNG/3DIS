# Workflow 子模組導覽（Step 4）

> `my3dis.workflow` 負責解析 YAML 設定、決定要執行哪些場景（scenes）、如何排程各個 stage（SSAM → Filter → Tracker → Report → 其他），並將結果彙整成結構化 summary 與歷史紀錄。本文件依實際執行順序說明各模組與主要函式。

## YAML 模式執行流程

```text
python -m my3dis.run_workflow --config <CONFIG>
    └─ main()
       ├─ _record_pid_map(config_path)
       └─ execute_workflow(config, override_output, config_path)
             ├─ 解析 experiment / stages（YAML 已透過 workflow.io.load_yaml 載入）
             ├─ scenes.normalize_scene_list() + 路徑展開
             ├─ （選配）初始化 OOM 監測
             └─ 逐場景執行：
                   ▼ run_scene_workflow(...)
                     └─ SceneWorkflow.run()
                        ├─ StageRecorder('ssam')
                        │   └─ my3dis.generate_candidates.run_generation()
                        │       └─ ssam_progressive_adapter.generate_with_progressive()
                        │           └─ progressive_refinement.progressive_refinement_masks()
                        ├─ StageRecorder('filter') [若啟用]
                        │   └─ my3dis.filter_candidates.run_filtering()
                        ├─ StageRecorder('tracker') [若啟用]
                        │   └─ my3dis.track_from_candidates.run_tracking()
                        │       └─ tracking.level_runner.run_level_tracking() × 每個 level
                        │           └─ tracking.sam2_runner.sam2_tracking()
                        ├─ StageRecorder('report') [若啟用]
                        │   └─ my3dis.generate_report.build_report()
                        └─ Finalise
                            ├─ summary.export_stage_timings()
                            ├─ summary.apply_scene_level_layout()
                            └─ summary.append_run_history()
```

---

## Step 4‑0 ─ 對外匯出介面（`__init__.py`、`core.py`）

### `workflow/__init__.py`
- 主要目的是提供穩定的 import 介面給 `run_workflow.py` 與其他模組使用，匯出：
  - `SceneContext`, `SceneWorkflow`, `run_scene_workflow`
  - `execute_workflow`
  - 常用工具：`StageRecorder`, `export_stage_timings`, `expand_output_path_template`,  
    `discover_scene_names`, `normalize_scene_list`, `resolve_levels`, `stage_frames_string`,  
    `resolve_stage_gpu`, `derive_scene_metadata`, `update_summary_config`, `load_manifest`,  
    `append_run_history`, `apply_scene_level_layout`, `load_yaml`, `using_gpu`,  
    `now_local_iso`, `now_local_stamp`, `build_completion_log_entry`, `log_completion_event`。

### `core.py`
- 作為向後相容層，將歷史上 `my3dis.workflow.core` 的匯出重新導向至新實作：
  - `SceneContext`, `SceneWorkflow`, `run_scene_workflow`, `execute_workflow` 皆來自目前的 `scene_workflow` 與 `executor`。

---

## Step 4‑1 ─ 錯誤型別、I/O 與一般工具

### `errors.py`
- `WorkflowError`：所有 workflow 相關錯誤的基底類別。
- `WorkflowConfigError`：設定檔無效時拋出，例如 YAML 結構錯誤或缺少必要欄位。
- `WorkflowRuntimeError`：執行過程中發生 I/O 或非預期副作用時拋出。

### `io.py`
- `load_yaml(path)`：
  - 以安全方式讀取 YAML；
  - 保證頂層為 mapping，否則拋出 `WorkflowConfigError`；
  - 訊息有助於使用者快速定位錯誤設定檔。

### `utils.py`
- GPU 設定相關：
  - `_collect_gpu_tokens(spec)`：將字串或序列型態的 GPU 指定轉換成乾淨 token list。
  - `normalise_gpu_spec(gpu)`：將多種 GPU 格式（例如 `"0,1"` 或 `[0,1]`）正規化為整數索引列表，去除重複與負值。
  - `serialise_gpu_spec(gpu)`：將索引列表序列化為逗號分隔字串，供 `CUDA_VISIBLE_DEVICES` 使用。
  - `using_gpu(gpu)`：
    - context manager，暫時設定 `CUDA_VISIBLE_DEVICES`；
    - 用於場景級執行時避免多個 process 互相覆蓋 GPU 配置。
- 時間：
  - `now_local_iso()`：回傳本地時區的 ISO 格式字串，用於 summary 與 log。
  - `now_local_stamp()`：產生適合檔名使用的時間戳（無冒號等特殊字元）。

---

## Step 4‑2 ─ 場景解析與路徑展開（`scenes.py`）

### `scenes.py`
- `expand_output_path_template(path_value, experiment_cfg)`：
  - 將 `experiment.output_root` 中的 `{name}` 模板替換成實際實驗名稱；
  - 若缺少對應欄位，會明確拋出設定錯誤。
- `discover_scene_names(dataset_root)`：
  - 在資料集根目錄下尋找 `scene_*` 形式的子資料夾；
  - 回傳排序後的場景名稱列表。
- `normalize_scene_list(raw_scenes, dataset_root, scene_start, scene_end)`：
  - 支援：
    - 明確列表（`['scene_00005_00', 'scene_00005_01']`）；
    - 單一字串（`'scene_00005_00'`）；
    - 特殊關鍵字 `all`；
    - 範圍過濾（`scene_start`/`scene_end`）。
  - 會同時檢查場景是否存在於檔案系統，避免靜默失敗。
- `resolve_levels(stage_cfg, manifest, fallback)`：
  - 解析各個 stage 實際應使用的 levels：
    1. 優先採用 stage 自己的設定（例如 `stages.tracker.levels`）；
    2. 若未指定則參考 SSAM `manifest` 中寫入的 levels；
    3. 再退回到 `experiment.levels`。
- `stage_frames_string(stage_cfg, experiment_cfg=None)`：
  - 將 `experiment.frames` 與各 stage override 合併，產生統一的 `start:end:step` 字串。
- `resolve_stage_gpu(stage_cfg, default_gpu)`：
  - 決定某 stage 實際使用的 GPU 配置；
  - 若 stage 未明確指定，則沿用 `experiment.gpu` 或 workflow 預設值。
- `derive_scene_metadata(data_path)`：
  - 從 `experiment.data_path` 推斷：
    - `scene` 名稱；
    - `scene_root`（場景根目錄）；
    - `dataset_root`（資料集根目錄）；
  - 這些欄位會被寫入 summary 中，方便後續分析／稽核。

---

## Step 4‑3 ─ 執行紀錄與 Summary 工具（`summary.py`、`logging.py`、`manifest_manager.py`、`reporting.py`）

### `summary.py`
- 系統環境快照：
  - `_gather_git_snapshot()`：在 repo 受 Git 管理時，收集當前 commit／branch／dirty 狀態；
  - `collect_environment_snapshot()`：
    - 彙整 Python 版本、平台資訊、CUDA、PyTorch、NumPy 版本與關鍵環境變數；
    - 輸出到 `environment_snapshot.json`。
- 資源監控：
  - `_bytes_to_mib(value)`：將 byte 轉成 MiB。
  - `_normalise_gpu_indices(spec)`：包裝 `normalise_gpu_spec` 供監控使用。
  - `StageResourceMonitor`：
    - `__init__(stage_name, gpu_spec, poll_interval)`：指定監控對象與輪詢間隔；
    - `start()`／`stop()`：啟動與停止監控執行緒；
    - 內部使用 psutil 與 CUDA API 收集 CPU/GPU 記憶體與使用率（透過 `_setup_cpu_monitor`、`_collect_process_rss`、`_update_cpu_peak`、`_poll_cpu_usage` 等）。
    - `_setup_gpu_monitor`／`_safe_cuda_call`／`_finalise_gpu_metrics`：在 GPU 可用時收集相關數據，並安全處理例外。
    - `_build_cpu_summary()`：將 CPU 指標轉換成易讀摘要。
- `StageRecorder`：
  - 以 context manager 形式整合 stage 執行、資源監控與 summary 記錄：
    - `__post_init__()`：正規化 GPU 設定；
    - `__enter__()`：登錄 stage 名稱與起始時間，啟動 `StageResourceMonitor`；
    - `__exit__(...)`：記錄結束時間、耗時、例外資訊與資源指標，將結果寫入 summary 的 `stages` 區段。
- 其他 Summary 工具：
  - `export_stage_timings(summary, output_path)`：將 per-stage timing 寫成獨立 JSON，供後續報表或外部工具使用。
  - `update_summary_config(summary, config)`：將原始 YAML 設定的 snapshot 嵌入 summary 中，方便完全重現實驗。
  - `load_manifest(run_dir)`：安全載入 `manifest.json`，若失敗則回傳 `None` 而非拋例外。
  - `_lock_file_handle(handle)`／`_unlock_file_handle(handle)`：跨平台檔案鎖，確保 run history 追加時具備基本原子性。
  - `append_run_history(summary, manifest, history_root)`：
    - 將當前場景執行資訊追加到 `logs/workflow_history.csv`；
    - 欄位包含：timestamp、scene 名稱、實驗名稱、config 路徑、data 路徑、output root、run_dir、levels、ssam_freq、幀數統計等。
  - `_move_file(src, dst)`：在調整輸出 layout 時，提供更安全的檔案搬移。
  - `_load_json_if_exists(path)`：若檔案存在則讀取 JSON，否則回傳 `None`。
  - `apply_scene_level_layout(run_dir, summary, manifest)`：
    - 根據 `experiment.output_layout` 重新整理輸出目錄結構；
    - 將 per-level 結果提升到較穩定的目錄配置；
    - 更新 manifest 與 summary 中的路徑欄位。

### `logging.py`
- `build_completion_log_entry(status, config_path, summary, message=None)`：
  - 建立適合寫入 append-only 日誌的結構化訊息：
    - 包含 config 路徑、場景清單、完成／失敗狀態、基礎統計。
- `log_completion_event(status, config_path, message)`：
  - 將完成事件附加到 `logs/workflow_notifications.log`；
  - 可由外部監控系統使用。

### `manifest_manager.py`
- `Manifest` dataclass：封裝 `manifest.json` 的讀寫與變更追蹤。
  - `Manifest.load(run_dir)`：
    - 從 run 目錄載入 manifest；
    - 若檔案不存在或無法解析，會警告並回傳空 dict。
  - `save(force=False)`：
    - 如果 `_modified` 標記為 True，或 `force=True`，則將 `data` 寫回 `manifest.json`；
    - 確保目錄存在並處理 I/O 例外。
  - `get(key, default=None)`／`set(key, value)`／`update(updates)`：
    - 提供類字典介面並自動維護 `_modified` 標記。
  - 常用屬性：
    - `levels`：回傳 `data['levels']`（若存在）；
    - `frames`：回傳 frames metadata；
    - `mask_scale_ratio`：將 `data['mask_scale_ratio']` 正規化為 float，預設 1.0。
  - Context manager：
    - `__enter__()`／`__exit__()`：
      - 在 `with Manifest.load(...) as m:` 區塊結束時，若無例外則自動呼叫 `save()`。

### `reporting.py`
- 資料結構：
  - `SceneStats`：
    - 紀錄每場景的物件／家族統計（`objects_per_level`、`total_objects`、`total_families`、`orphan_objects`、`max_depth` 等）以及每層級的 frame 數。
  - `SceneTiming`：
    - 描述某場景 summary 檔是否存在、總耗時、各 stage 耗時與對應之 `SceneStats`。
- 函式：
  - `format_seconds(seconds)`：將秒數格式化為 `H:MM:SS` 或 `M:SS`。
  - `load_scene_stats(scene_dir)`：
    - 優先讀取 `relations/family_tree.json`，若失敗則回退到 legacy `relations.json`；
    - 呼叫 `_load_stats_from_family_tree` 或 `_load_stats_from_legacy_relations` 解析統計。
  - `_load_stats_from_family_tree(family_tree, scene_dir)`：
    - 從 `statistics` 欄位讀取各層級物件數與家族數；
    - 透過 `_calculate_object_depth` 計算層級深度；
    - 由各 level 的 `index.json` 估計 frame 數。
  - `_load_stats_from_legacy_relations(relations, scene_dir)`：
    - 根據 `hierarchy` 與 `paths` 重建物件／家族統計，並透過 per-level `index.json` 補足 frame 數。
  - `_calculate_object_depth(obj_id, objects)`：
    - 沿著 `parent_id` 鏈往上追蹤，計算深度（root = 1）。
  - `load_scene_timing(scene_dir)`：
    - 讀取 `summary.json` 中的 timings；
    - 以 `SceneTiming` 包裝總耗時與 per-stage 耗時資訊；
    - 同時載入 `SceneStats` 作為補充。
  - `collect_stage_order(scenes)`：
    - 收集所有場景中出現過的 stage 名稱，建立統一欄位順序。
  - `render_markdown_table(stage_order, scenes)`：
    - 根據 `SceneTiming` 產生 per-stage 耗時的 Markdown 表格。
  - `render_stats_table(scenes)`：
    - 以 Markdown 呈現 family tree 統計（總物件、家族數、orphan 數、max depth、每層級幀數）。
  - `build_experiment_report(base_path)`：
    - 掃描 experiment 目錄中所有場景；
    - 以 `SceneTiming` 與 `SceneStats` 建立跨場景報告，回傳字串形式的 Markdown。
  - `generate_experiment_check_report(experiment_dir)`：
    - 呼叫 `build_experiment_report` 並寫入檔案；
    - 回傳報告檔路徑，以便 QA 或批次檢查使用。

---

## Step 4‑4 ─ 單場景工作流程（`scene_workflow.py` 與 `stage_runner.py`、`stage_config.py`、`stages/`）

### `stage_config.py`
- `SSAMStageConfig`：
  - 將 `stages.ssam` 中的所有設定（levels、frame range、`min_area`、`fill_area`、`stability_threshold`、`ssam_freq`、`max_masks_per_level`、mask 縮放、是否寫 raw 等）集中為 dataclass；
  - `from_yaml_config(stage_cfg, experiment_cfg, data_path, output_root)`：
    - 透過 `resolve_levels` 與 `stage_frames_string` 解析層級與幀範圍；
    - 驗證並解析 Semantic-SAM checkpoint 路徑（包含預設與錯誤訊息）；
    - 正規化所有整數與浮點參數，處理特殊值（例如 `max_masks_per_level <= 0` 表示無上限）。
  - `to_legacy_kwargs()`（在此類別中由 workflow 使用）：
    - 將 dataclass 轉換為 `generate_candidates.run_generation` 所期待的 kwargs。
- `TrackingStageConfig`：
  - 將 `stages.tracker` 設定封裝為 dataclass，包含：
    - SAM2 config／checkpoint 路徑；
    - levels、`sam2_max_propagate`、`iou_threshold`；
    - mask 縮放與 downscale 設定；
    - prompt mode（`none`／`long_tail`／`all`）與視覺化選項；
    - SAM2 tree 建立與可視化相關參數。
  - `from_yaml_config(stage_cfg, experiment_cfg, data_path, candidates_root, output_root, manifest)`：
    - 驗證路徑存在性；
    - 結合 SSAM manifest 中的 mask 縮放資訊，推導 tracker 的預設縮放；
    - 檢查 `prompt_mode` 是否在容許集合內，並轉成 `long_tail_box_prompt`／`all_box_prompt` 布林旗標。
- `FilterStageConfig`：
  - 包含 `root`、`levels`、`min_area`、`stability_threshold` 等；
  - `from_yaml_config(stage_cfg, experiment_cfg, candidates_root)`：
    - 使用 `resolve_levels` 決定實際濾波層級；
  - `to_legacy_kwargs()`：
    - 轉換成 `filter_candidates.run_filtering` 所需的 kwargs。

### `stage_runner.py`
- `StageRunner`（抽象基底類別）：
  - 屬性：
    - `context: SceneContext`：場景執行 context；
    - `summary: Dict[str, Any]`：workflow summary（由 SceneWorkflow 控制）。
    - `_stage_gpu_env: Optional[str]`：序列化後的 GPU 指定字串。
  - 介面：
    - 抽象 property `name`：各 stage 的識別名稱（`'ssam'`、`'filter'`、`'tracker'`、`'report'` 等）。
    - 抽象方法 `should_run()`：決定此 stage 是否應該實際執行。
    - 抽象方法 `execute()`：具體執行邏輯；需負責更新 `summary`。
  - 輔助方法：
    - `_stage_cfg()`：從 `context.stages_cfg` 中取出對應 stage 的設定區塊，若不存在則回傳空 dict。
    - `_stage_summary()`：在 `summary['stages']` 中建立／取得當前 stage 的 summary 子字典。
    - `set_gpu_env(gpu_env)`：告知 StageRunner 目前使用的 GPU 設定，便於記錄。

### `scene_workflow.py`
- `SceneContext`：
  - 封裝 YAML `config`、`experiment_cfg`、`stages_cfg`、預設 GPU 設定、data path、output root、config 路徑與 parent meta（用於多場景實驗索引）。
- `SceneWorkflow`：
  - `__init__(context)`：
    - 建立 summary 骨架，記錄 `config_path` 與 `invoked_at` 等；
    - 呼叫 `_populate_experiment_metadata()` 將場景／實驗資訊寫入 `summary['experiment']`；
    - 根據 `experiment.output_layout` 決定 layout 模式。
  - `run()`：
    - 在 `using_gpu(self.default_stage_gpu)` context 中執行：
      1. `_run_ssam_stage()`；
      2. `_run_filter_stage()`；
      3. `_run_tracker_stage()`；
      4. `_run_family_tree_stage()`（若存在；新版本使用 stage runner）；
      5. `_run_family_viz_stage()`（若啟用）；
      6. `_run_report_stage()`；
      7. `_finalize()`。
    - 每一 stage 中，通常會：
      - 從 `stages_cfg` 拿到該 stage 設定；
      - 建立對應的 `StageRunner` 或直接呼叫 CLI 包裝；
      - 使用 `StageRecorder` 記錄時間與資源；
      - 更新 `self.run_dir` 與 `self.manifest` 等屬性。
  - 其他關鍵方法（新實作與 legacy 版本在概念上相同）：
    - `_stage_cfg(name)`／`_stage_summary(name)`：類似於 StageRunner 中的輔助函式，但在沒有 StageRunner 版本時使用。
    - `_determine_layout_mode()`：解析 `output_layout`（例如 `flat`／`by_scene`）；
    - `_populate_experiment_metadata()`：結合 `derive_scene_metadata` 與 parent meta，建立實驗描述；
    - `_ensure_run_dir()`：確保 SSAM run 目錄存在（若 stage 設定為 reuse 模式）；
    - `_ensure_manifest()`：載入並快取 SSAM manifest。

### `scene_workflow_legacy.py`
- 保留 pre‑v1 的 `SceneContext`／`SceneWorkflow` 實作，供舊版腳本與長期實驗沿用：
  - 方法命名與新的 `SceneWorkflow` 大致一致（`_run_ssam_stage`、`_run_filter_stage` 等），但以純 dict 操作為主；
  - 暫時與新的 stage runner 架構共存，便於逐步遷移。

### `stages/`（具體 Stage Runner）
- `ssam_stage.py` — `SSAMStageRunner`：
  - `name` → `"ssam"`；
  - `should_run()`：由 `stages.ssam.enabled` 決定（預設 True）；
  - `execute()`：
    - 若 disabled，則呼叫 `_handle_reuse_existing()`，嘗試從 `experiment.run_dir` 或 output root 下既有 run 目錄中尋找可用的 `manifest.json`；
    - 否則呼叫 `_run_candidate_generation()`：
      - 使用 `SSAMStageConfig.from_yaml_config()` 驗證 YAML；
      - 透過 `StageRecorder` 呼叫 `generate_candidates.run_generation()`；
      - 將回傳的 `run_root` 與 `manifest` 寫入 `SceneContext` 與 summary。
- `filter_stage.py` — `FilterStageRunner`：
  - 執行 Stage 1b：重新對 SSAM 候選套用面積／穩定度濾波；
  - `execute()` 中呼叫 `filter_candidates.run_filtering()`，並將 per-level 統計寫入 summary。
- `tracker_stage.py` — `TrackerStageRunner`：
  - 執行 SAM2 追蹤 stage；
  - 使用 `TrackingStageConfig.from_yaml_config()` 解析設定；
  - 透過 `StageRecorder` 呼叫 `track_from_candidates.run_tracking()`；
  - 將 video/object segments 路徑、長尾框提示模式與統計寫入 summary。
- `family_tree_stage.py` — `FamilyTreeStageRunner`：
  - 根據 `family_tree` stage 設定決定是否執行；
  - 呼叫 `family_tree_builder.build_family_tree()`，並將 `family_tree.json` 路徑與統計寫入 summary。
- `family_viz_stage.py` — `FamilyVizStageRunner`：
  - 當 YAML 提供指定 family/物件清單時，載入 `FamilyTreeQuery`，產生子圖 JSON 或圖像，供 QA 或展示使用。
- `report_stage.py` — `ReportStageRunner`：
  - 呼叫 `generate_report.build_report()` 建立場景層級的 Markdown 報告；
  - 支援覆寫輸出檔名與圖片最大寬度。

---

## Step 4‑5 ─ 多場景排程與平行化（`executor.py`）

### `executor.py`
- `_SceneJob`（dataclass）：
  - 描述單一場景工作：場景索引、場景名稱與建立 `SceneContext` 所需的 kwargs。
- `_SceneProcessHandle`（dataclass）：
  - 封裝場景 process、通訊管道與 job。
- `_sanitise_scene_name(name)`：
  - 將場景名稱轉成適合檔名的形式（只保留英數字與 `-`／`_`）。
- `_prepare_scene_log(job)`：
  - 為每個場景建立獨立 log 檔：
    - 根據 config 名稱與 run timestamp 建立 log 目錄；
    - 建立 `logging.FileHandler` 與 `_TeeStream`，將 stdout/stderr 同時寫入檔案與原終端機。
- `_cleanup_scene_log(handler, stack)`：
  - 關閉 log handler 與 context stack，確保檔案描述符被釋放。
- `_run_scene_job(job)`：
  - 在目前 process 中執行單一場景：
    - 設定 entry log 格式；
    - 準備 log 檔；
    - 呼叫 `run_scene_workflow(**job.kwargs)`；
    - 將 scene log 路徑加入 summary 的 `experiment.scene_log`。
- `_scene_job_worker(conn, job)`：
  - 供新 spawn 出的 child process 使用；
  - 捕捉例外與 traceback，透過 pipe 回傳給 parent。
- `_launch_scene_process(job, ctx)`：
  - 建立 multiprocessing `Process`，以 `_scene_job_worker` 為目標；
  - 回傳 `_SceneProcessHandle`。
- OOM 監控與路徑覆寫：
  - `_resolve_path_override(env_var, configured)`：允許透過環境變數覆寫 YAML 中的資料根目錄或輸出根目錄；
  - `_prepare_memory_event_readers(paths)`／`_read_memory_snapshots(readers)`／`_detect_memory_events(readers, previous, current)`：
    - 透過 `oom_monitor.memory_events` 讀取 cgroup `memory.events` 檔案；
    - 偵測執行期間是否發生 OOM；
  - `_format_oom_events(events)`：將 OOM 事件格式化為可讀字串。
- `execute_workflow(config, override_output, config_path, memory_event_paths)`：
  - 整個 YAML 模式的主入口：
    1. 解析 experiment 與 stages；
    2. 透過 `normalize_scene_list` 決定場景列表；
    3. 視設定決定是否平行執行（多個 process）；
    4. 執行每個 `_SceneJob` 並收集 summary 與 OOM 相關資訊；
    5. 回傳整體執行結果與統計。

---

## Step 4‑6 ─ Workflow 附屬資料（`logs/`）

- `workflow/logs/`：
  - `workflow_history.csv`：
    - 由 `append_run_history` 寫入；
    - 包含每次場景執行的時間戳、場景名稱、experiment 名稱、config 路徑、資料來源、輸出根、run_dir、levels、SSAM 頻率與幀統計等。
  - `scene_workers/`（執行時動態建立）：
    - 每個場景在獨立 process 中執行時，stdout/stderr 會被 tee 到對應的 log 檔案中；
    - `SceneWorkflow` 的 summary 會紀錄 scene log 路徑，方便事後追蹤。
  - `workflow_notifications.log`：
    - `log_completion_event()` 將每次 workflow 完成／失敗事件追加至此，以便外部監控。

---

### 延伸閱讀

- My3DIS 全體模組與執行流程：`src/my3dis/OVERVIEW.md`
- Tracking 子系統：`src/my3dis/tracking/TRACKING_GUIDE.md`
- SSAM 候選產生與 cascade 濾波：`src/my3dis/generate_candidates.py`、`src/my3dis/cascade_filter.py`
- 3D 聚合與 SigLIP 指派：`src/my3dis/aggregation/README.md`、`src/my3dis/siglip_assignment/README.md`

