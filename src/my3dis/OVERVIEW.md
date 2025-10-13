# My3DIS 核心模組說明（依實驗流程）

> 本文件按「實驗流程」視角整理 `src/my3dis` 目錄下各個 `.py` 的角色與函式。  
> 追蹤與工作流程子模組請參照 `tracking/TRACKING_GUIDE.md`、`workflow/WORKFLOW_GUIDE.md` 取得更細節。

## 執行流程圖（總覽與模式分支）

```
                   ┌──────────────────────────────────────┐
                   │            執行模式選擇              │
                   └──────────────────────────────────────┘
                         │                     │
              YAML Orchestrator           Two-Stage CLI
               (建議：多場景)            (單場景/自訂)
                         │                     │
            python -m my3dis.run_workflow     ├─────────▶ SSAM 候選：
                         │                     │            my3dis.generate_candidates: run_generation
          ▼ execute_workflow(...)              │            ├─ ssam_progressive_adapter.generate_with_progressive
              for scene in scenes                  │            │   └─ semantic_refinement.progressive_refinement_masks
              ▼ SceneWorkflow.run()            │            └─ raw_archive.persist_raw_frame / filtered.json
              ├─ SSAM → generate_candidates    │
              ├─ Filter → filter_candidates    │          ▶ SAM2 追蹤：
              ├─ Tracker → track_from_candidates          my3dis.track_from_candidates: run_tracking
              ├─ Report → generate_report                 ├─ prepare_tracking_context / ensure_subset_video
              └─ Finalize → apply_scene_level_layout      ├─ for level: tracking.level_runner.run_level_tracking
                         │                                │    ├─ candidate_loader.iter_candidate_batches
                         │                                │    ├─ sam2_runner.sam2_tracking
                         │                                │    │   ├─ _prepare_prompt_candidates
                         │                                │    │   ├─ _filter_new_candidates (DedupStore)
                         │                                │    │   ├─ _add_prompts_to_predictor
                         │                                │    │   └─ _propagate_frame_predictions
                         │                                │    ├─ persist_level_outputs
                         │                                │    │   ├─ outputs.build_video_segments_archive
                         │                                │    │   └─ outputs.build_object_segments_archive
                         │                                │    └─ outputs.save_comparison_proposals (可選)
                         │                                └─ update_manifest
```

## Step 0：共用環境與基礎工具

### `__init__.py`
- `__version__`：若已安裝成套件，讀取發佈版本；在原始碼樹下則落在 `0.0.0`，僅供外部程式取得版本資訊。

### `common_utils.py`
- `ensure_dir(path)`：建立（含父層）資料夾並回傳絕對路徑，所有階段在輸出檔案前都會呼叫。
- `format_duration(seconds)`：把秒數轉為 `MM:SS` / `HH:MM:SS`，用於 log 與報告時間資訊。
- `encode_mask(mask)`、`pack_binary_mask(mask, full_resolution_shape)`：把布林遮罩壓成 JSON 友善格式，後續追蹤、報告會重複使用。
- `downscale_binary_mask(mask, ratio)`：以 box filter 方式降解析度，配合 `mask_scale_ratio` 加速追蹤與視覺化。
- `is_packed_mask(entry)`、`unpack_binary_mask(entry)`：辨識與還原壓縮遮罩，確保各階段讀寫一致。
- `numeric_frame_sort_key(fname)`：依檔名內的數字排序影格，避免字典序錯亂。
- `list_to_csv(values)`：把設定列表壓成逗號字串，寫 manifest／log 時使用。
- `parse_levels(levels)`：解析 CLI 或 YAML 中的層級設定為整數列表。
- `parse_range(range_str)`：把 `start:end:step` 轉為三元組，供影格抽樣。
- `bbox_from_mask_xyxy(mask)`、`bbox_xyxy_to_xywh(bounds)`：在遮罩與 SAM/SAM2 需求的 bbox 表示間轉換。
- `build_subset_video(frames_dir, selected, selected_indices, out_root, folder_name)`：把抽樣影格整理成子資料夾（優先建立 symlink），供 Semantic-SAM 與 SAM2 共用。
- `setup_logging(explicit_level, env_var, logger_names_to_quiet)`：統一設定 root logger，並可壓低外部套件噪音。
- `configure_entry_log_format(explicit_level)`：把 log 格式套用時間戳與 PID，入口腳本（如 `run_workflow.py`）使用。

### `pipeline_defaults.py`
- `_path_from_env(env_var, fallback)`：允許以環境變數覆蓋預設路徑且回傳實體 `Path`。
- `expand_default(path)`：把內建的 `Path` 轉成絕對字串，即使尚未存在也能指向預期位置。

## Step 1：Semantic-SAM 候選生成

### `generate_candidates.py`
- `_coerce_packed_mask(entry)`：把輸入的遮罩統一轉為 packed 格式，確保 metadata/NPZ 都能讀回。
- `_mask_to_bool(entry)`：把遮罩轉為布林陣列（若原本就為 packed 會同時還原）。
- `_coerce_union_shape(mask, target_shape)`：嘗試把遮罩拉伸成同一尺寸，供 gap-fill 或 union 計算使用。
- `persist_raw_frame(level_root, frame_idx, frame_name, candidates, chunk_writer)`：按照 pipeline 格式把單幀候選寫成 metadata + NPZ，提供之後重新過濾。
- `configure_logging(explicit_level)`：初始化此模組的 logger，並壓制 Semantic-SAM 自身輸出。
- `run_generation(...)`：整合抽樣影格、呼叫 `ssam_progressive_adapter`, 寫入 `level_*` 目錄、建立 manifest 與統計，是 Step 1 的主流程。
- `main()`：CLI 入口，解析參數後呼叫 `run_generation`。

### `ssam_progressive_adapter.py`
- `_semantic_sam_workdir()`：暫時切換至 Semantic-SAM repo 目錄，避免套件尋找失敗。
- `_extract_gap_components(segs, fill_area)`：從一組遮罩中找出未覆蓋的大區域，支援 gap-fill。
- `generate_with_progressive(frames_dir, selected_frames, sam_ckpt_path, levels, ...)`：包裝 `progressive_refinement_masks`，逐幀產生各層候選並補上 gap-fill 遮罩，最後以 packed 格式回傳。

### `progressive_refinement.py` ⚠️ **相容層**
- **僅保留舊匯入路徑**：會發出棄用警告，後續版本將移除。
- 重新導出 `semantic_refinement` / `semantic_refinement_cli` 的公開 API，舊代碼依然可呼叫 `progressive_refinement_masks()`、`setup_output_directories()` 等函式。

### `semantic_refinement.py` ⭐ **核心模組**
- `progressive_refinement_masks()`：主要漸進式細化演算法。
- `create_masked_image()`、`prepare_image_from_pil()`：生成 masked 圖、避免中途落盤。
- `setup_output_directories()`、`save_original_image_info()`：統一輸出結構與原圖備份。
- `bbox_from_mask()`、`instance_map_to_color_image()`：輔助函式與視覺檢查工具。

### `semantic_refinement_cli.py` ⭐ **CLI 模組**
- `parse_args()`：支援單張或整個場景批次、checkpoint 指定、gap-fill 參數等。
- `main()`：整合 logging、模型載入、manifest/index 生成，再呼叫核心演算法。

### `raw_archive.py`
- `_PendingFrame`：內部暫存單幀資料（metadata、遮罩二進位、統計），供 writer flush。
- `RawCandidateArchiveWriter`：將每幀原始候選打包成 chunked tar，避免大量檔案散落，可搭配 `with` 使用。
  - `__init__(level_root, chunk_size, compression)`：初始化輸出資料夾與緩衝設定。
  - `__enter__()` / `__exit__()`：支援 context manager，自動在離開時寫出剩餘緩衝。
  - `manifest_path`：屬性，回傳 manifest 的實際路徑。
  - `add_frame(frame_idx, frame_name, meta_bytes, mask_bytes, candidate_count, mask_count)`：累積單幀原始資料，達到 chunk 門檻時自動寫檔。
  - `close()`：寫出 manifest，清空緩衝。
  - 內部私有方法 `_make_chunk_name`、`_flush` 用於 chunk 檔名與真正落盤。
- `RawCandidateArchiveReader`：對應的讀取器，既能讀新格式（tar chunk），也能回退到舊版 JSON/NPZ。
  - `has_manifest()`：確認是否存在新版 chunk manifest。
  - `frame_indices()`：列出所有存在的幀索引。
  - `load_frame(frame_idx)`：還原單幀 metadata 與遮罩堆疊，供 `filter_candidates` 或其他後處理使用。
  - `_legacy_frame_indices`、`_legacy_load_frame`：舊版檔案結構的 fallback。

## Step 1b：候選再濾與整理

### `filter_candidates.py`
- `FilterStats`：簡單的統計累積器。
  - `add_frame(kept_count, dropped_count)`：更新幀數、保留與丟棄數量。
  - `to_dict()`：轉為可寫入 JSON 的字典。
- `bbox_from_mask(mask)`：重建 XYXY bbox，與上游 `progressive_refinement` 搭配。
- `filter_level(level_root, min_area, stability_threshold, verbose)`：讀取 raw archive、依條件重新過濾、輸出 `filtered.json`。
- `run_filtering(root, levels, min_area, stability_threshold, update_manifest, quiet)`：遍歷各層級執行 `filter_level`，並視需求更新 manifest。
- `parse_args()` / `main()`：提供重新過濾的 CLI。

## Step 2：SAM2 追蹤準備與執行

> 入口 `track_from_candidates.py` 以及詳細追蹤邏輯、請參考 `tracking/TRACKING_GUIDE.md`。

### `track_from_candidates.py`
- `resolve_sam2_config_path(config_arg)`：將 CLI 指定的 config 轉為 SAM2 可識別的 Hydra 路徑或絕對檔案。
- `configure_logging(explicit_level)`：設定此模組 logger。
- `run_tracking(...)`：整合 manifest、載入 SAM2 predictor、逐層呼叫 `tracking.level_runner.run_level_tracking` 並統整成果，是 Step 2 的主流程。
- `main()`：SAM2 單獨階段 CLI 入口。

## Step 3：報告與視覺化

### `generate_report.py`
- `StageTiming`：封裝單一 stage 的耗時資訊，`duration_text` 屬性會轉成易讀格式。
- `load_json(path)`：安全讀取 JSON（容錯）。
- `collect_stage_timings(summary)`：從 workflow summary 抽出各 stage 的時間紀錄並排序。
- `pick_first_mid_last(items)`：挑選首／中／尾三張代表圖，避免報告過長。
- `downscale_image(src, dst, max_width)`：調整預覽圖尺寸並輸出。
- `render_level_section(level, viz_dir, report_dir, max_width, markdown_root)`：為每個 level 建立 Markdown 表格與對應圖像。
- `build_report(run_dir, report_name, max_preview_width)`：整合 manifest、summary、視覺化輸出完整報告，是 Step 3 的主要函式。
- `parse_args()` / `main()`：CLI 入口，可對任一 run 產生報告。

## Step 4：Workflow 管線（多場景協調）

> 詳細的 workflow 模組、場景切換、平行化與摘要請見 `workflow/WORKFLOW_GUIDE.md`。  
> 此處說明入口腳本：

### `run_workflow.py`
- `_configure_entry_logging()`：保證 CLI log 帶有時間與 PID。
- `_stdout_descriptor()`：偵測標準輸出導向（檔案、pipe）並回傳字串，寫入 PID map。
- `_record_pid_map(config_path)`：在 `logs/run_pid_map.csv` 新增一列，記錄本次執行細節。
- `main()`：解析 `--config` 等參數（含 `--dry-run` 用於僅檢查配置）、載入 YAML、呼叫 `workflow.execute_workflow`，並透過 `oom_monitor.memory_watch_context` 監控 OOM、寫入通知與歷史紀錄。

## Step 5：附屬工具

### `prepare_tracking_run.py`
- `build_destination(source, dest_root, name)`：決定複製目標資料夾名稱，支援自訂或 timestamp。
- `clone_directory(source, dest, mode)`：以 hardlink 或深度複本複製整個 run。
- `update_config(config_path, new_run_dir)`：把 YAML 裡的 `experiment.run_dir` 改指向新位置。
- `parse_args()` / `main()`：提供 CLI，用於重跑 SAM2 或建立備份。

---

- 追蹤模組詳細對照：`tracking/TRACKING_GUIDE.md`  
- Workflow 佇列與多場景執行細節：`workflow/WORKFLOW_GUIDE.md`
