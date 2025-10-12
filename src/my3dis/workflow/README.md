# Workflow 子模組說明（對應 Step 4）

> 本目錄負責解析 YAML、挑選場景、依序或平行執行各 stage，並記錄結果。  
> 下列內容依「設定解析 → 場景選擇 → 單場景執行 → 多場景排程 → 摘要輸出」排列，涵蓋每個函式/方法的角色。

## 執行流程圖（YAML 模式）

```
python -m my3dis.run_workflow --config <YAML>
    └─ main()
       ├─ _record_pid_map(config)
       └─ execute_workflow(config, override_output, config_path)
             ├─ 解析 experiment/stages（io.load_yaml 於上層已完成）
             ├─ scenes.normalize_scene_list() + 路徑展開
             ├─ （可選）記憶體事件監控初始化
             └─ for scene in scenes:
                   ▼ run_scene_workflow(...)
                     └─ SceneWorkflow.run()
                        ├─ StageRecorder('ssam')
                        │   └─ my3dis.generate_candidates.run_generation()
                        │       └─ ssam_progressive_adapter.generate_with_progressive()
                        │           └─ progressive_refinement.progressive_refinement_masks()
                        ├─ StageRecorder('filter') [若 enabled]
                        │   └─ my3dis.filter_candidates.run_filtering()
                        ├─ StageRecorder('tracker') [若 enabled]
                        │   └─ my3dis.track_from_candidates.run_tracking()
                        │       └─ tracking.level_runner.run_level_tracking() × N level
                        │           └─ tracking.sam2_runner.sam2_tracking()
                        ├─ StageRecorder('report') [若 enabled]
                        │   └─ my3dis.generate_report.build_report()
                        └─ Finalize
                            ├─ summary.export_stage_timings()
                            ├─ summary.apply_scene_level_layout()
                            └─ summary.append_run_history()
```

## Step 4-0：對外匯出

### `__init__.py`
- 將核心類別與工具（`SceneWorkflow`, `execute_workflow`, ...）重新匯出，提供上層 `run_workflow.py` 使用。

### `core.py`
- 單純 re-export `SceneContext`, `SceneWorkflow`, `run_scene_workflow`, `execute_workflow`，維持舊版本介面的相容性。

## Step 4-1：設定與錯誤類型

### `errors.py`
- `WorkflowError`：Workflow 相關錯誤的基底類別。
- `WorkflowConfigError`：配置檔有誤時拋出。
- `WorkflowRuntimeError`：IO/檔案操作等執行期問題。

### `io.py`
- `load_yaml(path)`：讀取 YAML 並保證最外層是 mapping，若找不到或格式錯誤會丟 `WorkflowConfigError`。

### `utils.py`
- `_collect_gpu_tokens(spec)`：解析各種格式的 GPU 指定字串／序列，轉成清洗後的 token 列表。
- `normalise_gpu_spec(gpu)`：把 GPU specification 轉成整數 index 列表，過濾重複與負值。
- `serialise_gpu_spec(gpu)`：把 GPU index 列表轉為逗號字串，供環境變數使用。
- `using_gpu(gpu)`：context manager，暫時設定 `CUDA_VISIBLE_DEVICES`，stage 執行時避免彼此干擾。
- `now_local_iso()` / `now_local_stamp()`：回傳本地時區的 ISO 與檔名友善時間戳。

## Step 4-2：場景解析與路徑展開

### `scenes.py`
- `expand_output_path_template(path_value, experiment_cfg)`：展開 `experiment.output_root` 中的 `{name}` 模板，若缺名稱會報錯。
- `discover_scene_names(dataset_root)`：列出 dataset 根目錄下的場景資料夾（偏好 `scene_*` 命名）。
- `normalize_scene_list(raw_scenes, dataset_root, scene_start, scene_end)`：處理 YAML 中的場景選擇（單一值、列表、`all` token、起迄範圍），並驗證存在性。
- `resolve_levels(stage_cfg, manifest, fallback)`：決定各 stage 要跑哪些層級，優先讀 stage 設定，其次 manifest，再次 fallback。
- `stage_frames_string(stage_cfg, experiment_cfg=None)`：合併 experiment 預設與 stage 覆寫，產出 `start:end:step` 字串。
- `resolve_stage_gpu(stage_cfg, default_gpu)`：回傳 stage 指定的 GPU，沒有就落回共用設定。
- `derive_scene_metadata(data_path)`：從資料路徑推導場景名稱、根目錄，寫入 summary。

## Step 4-3：執行紀錄與摘要

### `summary.py`
- `_bytes_to_mib(value)`：協助把位元組換算為 MiB。
- `_normalise_gpu_indices(spec)`：包裝 `normalise_gpu_spec` 供資源監視器使用。
- `_gather_git_snapshot()`：若有 git，擷取目前 commit/branch/dirty 狀態。
- `collect_environment_snapshot()`：收集 Python、平台、CUDA、torch、numpy 等環境資訊，寫入 summary。
- `StageResourceMonitor`：背景執行緒，紀錄 CPU/GPU 資源峰值。
  - `__init__(stage_name, gpu_spec, poll_interval)`：初始化監控設定。
  - `start()`：啟動 psutil/turbo 採樣與 GPU 記憶體記錄。
  - `stop()`：停止採樣並回傳資源統計。
  - `_setup_cpu_monitor()`、`_collect_process_rss()`、`_update_cpu_peak()`、`_poll_cpu_usage()`：處理 CPU 記憶體採樣與峰值統計。
  - `_setup_gpu_monitor()`、`_safe_cuda_call(fn, device)`、`_finalise_gpu_metrics()`：取得 CUDA 裝置資訊並統計峰值記憶體。
  - `_build_cpu_summary()`：整理 CPU 監控數據。
- `StageRecorder(summary, name, gpu)`：使用 context manager 方式記錄每個 stage 的起訖時間、資源資訊。
  - `__post_init__()`：正規化 GPU 設定。
  - `__enter__()`：註冊 stage 順序、起始時間並啟動資源監視器。
  - `__exit__(exc_type, exc, tb)`：記錄結束時間、耗時、錯誤與資源統計。
- `export_stage_timings(summary, output_path)`：把 stage 時間紀錄輸出成 JSON，供報告或外部分析。
- `update_summary_config(summary, config)`：把原始 config 快照附加在 summary 內。
- `load_manifest(run_dir)`：安全載入 `manifest.json`，失敗時回傳 `None`。
- `_lock_file_handle(handle)` / `_unlock_file_handle(handle)`：跨平台檔案鎖定，避免多程序同時寫歷史紀錄。
- `append_run_history(summary, manifest, history_root)`：把本次執行紀錄追加到 `logs/workflow_history.csv`。
- `_move_file(src, dst)`：移動檔案並處理錯誤，供整理輸出時使用。
- `_load_json_if_exists(path)`：讀取 JSON（若不存在則 `None`），用於蒐集 `stage_timings.json`。
- `apply_scene_level_layout(run_dir, summary, manifest)`：整理每個 level 的輸出（搬移 tracking 檔案、報告圖像、比較圖），重新寫回 manifest 與 summary。

### `logging.py`
- `build_completion_log_entry(status, config_path, started_at, finished_at, duration_seconds, run_summaries, error_message, traceback_text)`：把最終狀態組成可寫入檔案的文字，包含場景與錯誤摘要。
- `log_completion_event(status, config_path, message)`：把完成訊息附加到 `logs/workflow_notifications.log`。

## Step 4-4：單場景執行核心

### `scene_workflow.py`
- `SceneContext`：封裝單場景執行需要的原始設定、路徑、上層 metadata。
- `SceneWorkflow`
  - `__init__(context)`：解析 context、建立 summary 骨架與輸出配置。
  - `run()`：依序呼叫四個 stage（SSAM → Filter → Tracker → Report）並回傳 summary。
  - `_stage_cfg(name)`：取得 stage 子設定。
  - `_stage_summary(name)`：在 summary 內建立/取得 stage 統計欄位。
  - `_determine_layout_mode()`：檢查 `experiment.output_layout`。
  - `_populate_experiment_metadata()`：填入場景/資料路徑等資訊，含多場景時的父設定。
  - `_ensure_run_dir()`：確認 SSAM 階段產生的 run 目錄存在。
  - `_ensure_manifest()`：載入（或快取） SSAM 階段留下的 manifest。
  - `_run_ssam_stage()`：處理候選生成（可重用既有 run 或直接呼叫 `generate_candidates.run_generation`）。
  - `_run_filter_stage()`：若啟用則呼叫 `filter_candidates.run_filtering` 重新篩選。
  - `_run_tracker_stage()`：依設定呼叫 `track_from_candidates.run_tracking`。
  - `_run_report_stage()`：視需求建立報告與補充 summary。
  - `_finalize()`：整理輸出版型 (`apply_scene_level_layout`)、紀錄歷史 (`append_run_history`)。
- `run_scene_workflow(**kwargs)`：簡化呼叫介面，建立 `SceneWorkflow` 並執行 `run()`。

## Step 4-5：多場景排程與平行化

### `executor.py`
- `_SceneJob`：`dataclass`，描述單一場景工作（序號、場景名稱、建構 `SceneContext` 所需引數）。
- `_run_scene_job(job)`：在主程序中執行一個場景（主要給 thread/process pool 使用）。
- `_scene_job_worker(conn, job)`：spawn 出的子程序入口，執行場景並把結果（或錯誤資訊）透過 pipe 傳回。
- `_run_scene_job_isolated(job)`：建立獨立程序、等待完成、解析結果，回傳 `(success, payload)`。
- `_resolve_path_override(env_var, configured)`：允許環境變數覆蓋設定（dataset/output root）。
- `_prepare_memory_event_readers(paths)`：載入 `memory.events` 監控檔案，確定哪些可用。
- `_read_memory_snapshots(readers)`：讀取目前 OOM counters。
- `_detect_memory_events(readers, previous, current)`：找出 OOM 計數是否增加。
- `_format_oom_events(events)`：把 OOM 事件整理成字串，用於 log。
- `execute_workflow(config, override_output, config_path, memory_event_paths)`：主要進入點，解析 YAML → 決定場景清單 → 團隊執行（串行或平行）→ 整理回傳 summary 列表，並附上 OOM 偵測結果與時間戳。

## Step 4-6：輔助資料

- `logs/`：這個資料夾下方的 `workflow_history.csv` 由 `append_run_history` 追加，無程式碼。

---

如需返回「所有模組總覽」，請參考 `src/my3dis/README.md`。  
若要瞭解 SAM2 追蹤內部運作，請閱讀 `tracking/README.md`。
