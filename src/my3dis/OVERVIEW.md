# My3DIS 核心模組總覽（Execution Flow）

> 本文件依照實際執行順序，說明 `src/my3dis` 底下所有主要模組與其函式／類別職責，以及彼此之間的互動方式。  
> 追蹤子系統與 YAML 工作流程的更細節說明，請分別參考 `tracking/TRACKING_GUIDE.md` 與 `workflow/WORKFLOW_GUIDE.md`。

## 執行模式總覽

```text
                   ┌──────────────────────────────────────┐
                   │              執行模式                │
                   └──────────────────────────────────────┘
                         │                     │
              YAML 工作流程（多場景）      兩階段 CLI（單場景 / 即席）
                         │                     │
            python -m my3dis.run_workflow      ├─────▶ SSAM 候選產生
                         │                     │          generate_candidates.run_generation
          ▼ execute_workflow(...)              │          ├─ ssam_progressive_adapter.generate_with_progressive
              for scene in scenes              │          │   └─ semantic_refinement.progressive_refinement_masks
              ▼ SceneWorkflow.run()            │          └─ raw_archive.persist_raw_frame / filtered.json
              ├─ SSAM → generate_candidates    │
              ├─ Filter → filter_candidates    │          ▶ SAM2 追蹤：
              ├─ Tracker → track_from_candidates         track_from_candidates.run_tracking
              ├─ Report → generate_report                ├─ prepare_tracking_context / ensure_subset_video
              └─ Finalise → apply_scene_level_layout     ├─ level_runner.run_level_tracking（逐層）
                         │                               │    ├─ candidate_loader.iter_candidate_batches
                         │                               │    ├─ sam2_runner.sam2_tracking
                         │                               │    │   ├─ _prepare_prompt_candidates
                         │                               │    │   ├─ _filter_new_candidates（DedupStore）
                         │                               │    │   ├─ _add_prompts_to_predictor
                         │                               │    │   └─ _propagate_frame_predictions
                         │                               │    ├─ persist_level_outputs
                         │                               │    │   ├─ outputs.build_video_segments_archive
                         │                               │    │   └─ outputs.build_object_segments_archive
                         │                               │    └─ outputs.save_comparison_proposals（選配）
                         │                               └─ update_manifest
```

---

## Step 0 ─ 共用環境與基礎工具

### `__init__.py`
- 匯出套件版本號 `__version__`。  
  若以 pip 安裝，會回報實際版本；若直接從原始碼執行，則回退為 `"0.0.0"`。

### `common_utils.py`（相容層）

> 注意：此模組標記為 **DEPRECATED**，保留主要是為了舊版程式與腳本的相容性。新的程式碼應直接改用 `mask/`、`io/`、`utils/` 等專門模組。

- 透過 `from my3dis.mask.encoding import ...`、`from my3dis.mask.geometry import ...`、`from my3dis.io.fs_utils import ...` 等方式，**轉出口罩編碼、幾何計算、檔案系統操作與時間／解析度相關的工具函式**。
- 主要轉出的符號包括：
  - 遮罩編碼：`normalize_shape_tuple`, `encode_mask`, `pack_binary_mask`, `is_packed_mask`, `unpack_binary_mask`, `downscale_binary_mask`
  - 幾何工具：`bbox_from_mask_xyxy`, `bbox_xyxy_to_xywh`
  - 檔案系統：`ensure_dir`, `build_subset_video`
  - 解析與排序：`parse_levels`, `parse_range`, `list_to_csv`, `numeric_frame_sort_key`
  - 日誌與時間：`setup_logging`, `configure_entry_log_format`, `format_duration`
- 新開發建議直接從對應子模組匯入，例如 `from my3dis.mask.encoding import pack_binary_mask`。

### `mask/encoding.py`
- 常數：
  - `PACKED_MASK_KEY`, `PACKED_MASK_B64_KEY`, `PACKED_SHAPE_KEY`, `PACKED_ORIG_SHAPE_KEY`：標準化儲存「壓縮遮罩」時使用的欄位名稱，供 JSON／NPZ 共用。
- 函式：
  - `normalize_shape_tuple(shape)`：將 `ndarray`／list／tuple／int 等各種 shape 表示統一轉為 `Tuple[int, ...]`。
  - `encode_mask(mask)`：將布林遮罩壓縮成 base64 bitstream，方便寫入純 JSON 報表。
  - `pack_binary_mask(mask, full_resolution_shape=None)`：將遮罩打包成 compact dict，供 NPZ 與追蹤模組使用。
  - `is_packed_mask(entry)`／`unpack_binary_mask(entry)`：自動判斷／還原壓縮遮罩為布林陣列；若輸入本身就是 `ndarray`，則直接轉型。
  - `downscale_binary_mask(mask, ratio)`：以 box filter + 閾值法縮小遮罩，維持連通性。`generate_candidates` 和 `tracking` 的 downscale 都透過此函式。

### `mask/geometry.py`
- `bbox_from_mask_xyxy(mask)`：由遮罩中非零像素計算 `(x_min, y_min, x_max, y_max)`。
- `bbox_xyxy_to_xywh(bounds)`：將 XYXY 轉為 `(x, y, w, h)` 形式，對齊 SAM／SAM2 介面需求。

### `io/fs_utils.py`
- `ensure_dir(path)`：建立（含父層）目錄後回傳絕對路徑字串；幾乎所有產出檔案寫入前都會呼叫。
- `build_subset_video(frames_dir, selected, selected_indices, out_root, folder_name)`：
  - 依照選出幀列表，在輸出目錄下建立 `selected_frames/` 之類的子資料夾；
  - 優先建立 symlink，檔案系統不支援時改用複製；
  - 回傳 `(subset_dir, {abs_idx: subset_filename})`，供 SAM2 追蹤與可視化工具使用。

### `utils/` 系列模組
- `utils.logging`：
  - `setup_logging(explicit_level=None, env_var='MY3DIS_LOGLEVEL', logger_names_to_quiet=())`：統一設定 logging 格式與層級，並靜音過於冗長的第三方 loggers。
  - `configure_entry_log_format()`：在 entrypoint（例如 `run_workflow.py`）啟動時，套用包含 PID 與時間戳的 log 格式。
- `utils.parsing`：
  - `list_to_csv(values)`：清單轉 `a,b,c` 字串，用於 manifest 與 CSV 紀錄。
  - `parse_levels(level_spec)`：將 `2,4,6` 或 `[2,4,6]` 等輸入正規化成 `List[int]`。
  - `parse_range(range_str)`：解析 `start:end:step` 字串，回傳對應的 `(start, end, step)` 三元組。
- `utils.sorting`：
  - `numeric_frame_sort_key(filename)`：解析檔名中的數字索引（如 `frame_00123.png`），避免純字典序導致順序錯亂。
- `utils.time_utils`：
  - `format_duration(seconds)`：將秒數格式化為 `M:SS` 或 `H:MM:SS`，供 workflow summary 與報表使用。
- `utils.ply_utils`：
  - `_resolve_scene_ply(path)`／`load_scene_pointcloud(scene_root)`：從場景根目錄解析並載入點雲檔。
  - `save_mask_as_ply(points, mask, output_path)`：將某個遮罩對應的點集合輸出成 `.ply`，常用於手動檢查 3D proposal。

### `pipeline_defaults.py`
- `_path_from_env(env_var, fallback)`：讀取環境變數覆寫模型路徑／資料根目錄等設定，若未定義則回退到 repo 內建相對路徑。
- `_PROJECT_ROOT`：推斷專案根目錄，確保在不同工作目錄下執行仍能找到預設資源。
- 預設路徑常數：
  - `DEFAULT_SEMANTIC_SAM_ROOT`, `DEFAULT_SEMANTIC_SAM_CKPT`
  - `DEFAULT_SAM2_ROOT`, `DEFAULT_SAM2_CFG`, `DEFAULT_SAM2_CKPT`
  - `DEFAULT_DATA_PATH`, `DEFAULT_OUTPUT_ROOT`
- `expand_default(path)`：將 `Path` 轉成絕對路徑字串，即使實體檔案或資料夾尚未存在亦可使用。

### `cascade_filter.py`
- `CascadeFilter` 類別：
  - `__init__(min_area=None, stability_threshold=None)`：設定遮罩面積與穩定度閾值。
  - `filter_masks(masks)`：
    1. 由 `unique_id` 與 `parent_unique_id` 建立樹狀結構；
    2. 先對各節點個別套用面積／穩定度篩選；
    3. 對於被拒絕的父節點，將其所有子孫節點一併拒絕（cascade）；
    4. 回傳通過者與拒絕理由字典（`unique_id → reason`）。
  - `get_statistics(total_masks, rejection_reasons)`：計算各類拒絕原因的統計（面積、穩定度、cascade）。
- `cascade_filter_masks(masks, min_area, stability_threshold)`：方便用於一次性呼叫的包裝函式。
- `validate_parent_child_consistency(masks)`：檢查 `parent_unique_id` 是否都能在輸入集合中找到對應節點，避免形成孤兒。

### `torch_compat.py`
- `patch_torch_compat()`：在 `run_workflow` 啟動時呼叫一次，注入兩項相容性修補：
  - `_patch_torch_load_weights_only()`：於舊版 PyTorch 上新增 `weights_only` 參數的容忍，讓 SAM2 原始程式可直接呼叫 `torch.load(..., weights_only=True)`。
  - `_patch_scaled_dot_product_attention()`：若 `torch.nn.functional` 尚未提供 `scaled_dot_product_attention`，則以純 PyTorch 實作注入，避免強制升級 CUDA／PyTorch。

---

## Step 1 ─ Semantic-SAM 候選產生（SSAM）

此階段負責從 2D 影像產生多層級（例如 L2/L4/L6）Semantic-SAM 遮罩候選，並以結構化格式寫入磁碟，供後續濾波與 SAM2 追蹤使用。

### `generate_candidates.py`
- 內部工具：
  - `_coerce_packed_mask(entry)`：將輸入的遮罩（可能為 bool array、packed dict 等）統一轉成標準 packed 形式，並確保 `PACKED_SHAPE_KEY` 為 tuple[int]。
  - `_mask_to_bool(entry)`：確保遮罩以布林陣列呈現，必要時呼叫 `unpack_binary_mask`。
  - `_coerce_union_shape(mask, target_shape)`：將遮罩重採樣到指定尺寸，用於 gap-fill 運算等需要布林 union 的情境。
  - `_validate_and_prepare_params(fill_area, min_area, ssam_freq, mask_scale_ratio, downscale_masks)`：檢查 CLI 參數並回傳正規化後的 `(fill_area, ssam_freq, mask_scale_ratio)`。
  - `_select_frames(frames_dir, frames, ssam_freq)`：解析 frame range，挑選實際需處理的幀，並決定哪些幀要送入 Semantic-SAM。
  - `_build_output_folder_name(...)`：根據 level、`ssam_freq`、`sam2_max_propagate`、面積閾值、mask 縮放與實驗 tag 等資訊產生 run 目錄名稱。
  - `_detect_scene_metadata(frames_path)`：從影像路徑向上尋找 `scene_*` 目錄，推斷 `scene_name`、`scene_root` 與 `dataset_root`。
- `persist_raw_frame(level_root, frame_idx, frame_name, candidates, chunk_writer=None)`：
  - 將單一 frame 的候選遮罩切分為：
    - JSON metadata（不含 segmentation）；
    - NPZ 中的 packed mask stack；
  - 支援 legacy「一幀一檔」與新的 chunked tar 形式。
- `configure_logging(explicit_level)`：設定本模組的 logger，並關閉 Semantic-SAM 過於冗長的輸出。
- `run_generation(...)`：
  - 驗證與整理所有參數；
  - 透過 `_select_frames` 產生候選 frame 列表；
  - 呼叫 `ssam_progressive_adapter.generate_with_progressive` 取得逐幀逐層候選；
  - 依 level 建立 `level_{L}/` 目錄，寫入 raw／filtered／manifest 等檔案；
  - 回傳 `(run_dir, manifest_dict)`，供 workflow 與後續階段使用。
- `main()`：對應 CLI `python -m my3dis.generate_candidates`，是 YAML workflow `ssam` stage 與單獨執行時的入口。

### `ssam_progressive_adapter.py`
- `_semantic_sam_workdir()`：暫時切換工作目錄到 Semantic-SAM 專案根，確保其相對路徑 import 正常；完成後再切回原本 CWD。
- `_save_progressive_relations(all_results, save_root, levels)`：
  - 從 progressive refinement 的輸出中擷取 parent-child 關係；
  - 呼叫 `relation_index.save_ssam_relations` 將其存成 v3 `relations.json` 與 `tree_L*.json`。
- `_extract_gap_components(segs, fill_area)`：
  - 由一批 base-level 遮罩計算 coverage map；
  - 對未覆蓋區域進行連通元件分割，篩選出面積 ≥ `fill_area` 的 gap，回傳布林遮罩列表。
- `generate_with_progressive(frames_dir, selected_frames, sam_ckpt_path, levels, ...)`：
  - 於 `_semantic_sam_workdir` 區塊中建立 Semantic-SAM 模型；
  - 逐幀呼叫 `progressive_refinement_masks`，產生跨層級遮罩；
  - 若啟用 gap-fill，為 base-level 新增帶有 `unique_id`／`parent_unique_id` 等 provenance 的額外遮罩；
  - 將遮罩壓縮為 packed 形式，同時計算 bbox、area、stability 與縮放比；
  - 以 generator 形式回傳 `(frame_idx, frame_name, {level: [candidate_dicts]})`，供 `generate_candidates` 寫入磁碟；
  - 最後視需求寫出 SSAM `tree_L*.json` 與 merged `tree.json`。

### `semantic_refinement.py`（核心演算法）

> 此模組為原 `progressive_refinement.py` 的演算法核心抽取版本，可被 CLI、adapter、workflow 共用。

- 環境與模型：
  - 組態變數 `DEFAULT_SEMANTIC_SAM_ROOT` 決定 Semantic-SAM 原始碼位置。
  - 若無法成功 `import semantic_sam`，會建立一組 **lazy fallback** 函式，於使用時丟出資訊清楚的 `ImportError`。
- 文字輸出與計時：
  - `console(message, important=False)`：在 `MY3DIS_PROGRESSIVE_VERBOSE=1` 或 `important=True` 時輸出訊息。
  - `timer_decorator(func)`：裝飾器，用於記錄長耗時步驟的時間。
  - `log_step(step_name, start_time=None)`：標準化「開始／完成」訊息。
- 實驗管理：
  - `get_experiment_timestamp()`：產生 `YYYYmmdd_HHMMSS` 格式的時間戳。
  - `create_experiment_folder(base_path, experiment_name, timestamp=None)`：建立實驗資料夾並回傳 (path, timestamp)。
  - `setup_output_directories(experiment_path)`：建立 `original/`, `levels/`, `relations/`, `summary/`, `viz/` 子目錄。
  - `save_original_image_info(image_path, output_dirs)`：將輸入影像複製到 `original/` 並留下基本中繼資訊。
- 視覺輔助：
  - `prepare_image_from_pil(image)`／`create_masked_image(image, mask, color)`：生成疊加遮罩的預覽影像。
  - `bbox_from_mask(mask)`：從實例遮罩求出 bounding box。
  - `instance_map_to_color_image(instance_map)`：將 instance map 轉成 RGB 影像（若 Semantic-SAM 內建工具不可用，則使用 fallback 實作）。
- 主要流程：
  - `progressive_refinement_masks(semantic_sam, image_path, level_sequence, ...)`：
    - 依照多層級設定重複呼叫 Semantic-SAM；
    - 對每層進行面積與穩定度篩選，並透過 `CascadeFilter` 進行 parent-aware cascade filtering；
    - 保留完整的 `unique_id`／`parent_unique_id`／`lineage` 等 provenance 資訊，供 SAM2 與 family tree 使用；
    - 回傳結構化結果：`{'levels': {L: {'masks': [...]}}}` 與輔助統計。

### `semantic_refinement_cli.py` 與 `progressive_refinement_cli.py`
- `semantic_refinement_cli.py`：
  - `parse_args(argv)`：解析場景／單張影像模式、層級、框選範圍、面積閾值與 checkpoint 等。
  - `_resolve_image_paths(args)`：根據 `--scene` 與 `--frames` 或 `--image-path` 產生實際影像清單，並檢查邊界。
  - `_load_model(args)`：建立 Semantic-SAM 模型並輸出使用者友好的載入訊息。
  - `_write_manifest(experiment_path, metadata)`：寫入 `summary/manifest.json`，紀錄場景、層級與時間戳等。
  - `_update_index(index_csv, rows)`：維護跨實驗的索引檔（CSV）。
  - `main(argv)`：CLI 入口，調用上述輔助後執行 `progressive_refinement_masks`。
- `progressive_refinement_cli.py`：
  - 為舊路徑的相容層：重新匯出 `semantic_refinement_cli.main` 與 `parse_args`，並於 import 時顯示棄用警告。

### `raw_archive.py`
- `_PendingFrame` dataclass：暫存單一 frame 之 meta／mask bytes／統計資訊，等待 flush 進 tar chunk。
- `RawCandidateArchiveWriter`：
  - 使用 `chunk_size` 控制多少 frame 打包成一個 tar（可選 gzip/bz2/xz 壓縮）；
  - `add_frame(...)`：將 frame meta/mask bytes 與對應條目名稱佇列化；
  - `_flush()`：實際寫入 tar 檔並更新 manifest 中的 frame/chunk 索引；
  - `close()`／context manager：寫出最終 `manifest.json` 並清空 buffer。
- `RawCandidateArchiveReader`：
  - 支援兩種格式：
    1. 新式 chunked tar + manifest；
    2. 舊式 per-frame JSON + NPZ 檔案。
  - `frame_indices()`：列出所有可用 frame 索引。
  - `load_frame(frame_idx)`：讀回 `{'meta', 'mask_stack', 'packed_masks', 'mask_shape', 'has_mask'}` 結構，供 `filter_candidates` 等下游使用。

---

## Step 1b ─ 候選重新濾波（Filter）

### `filter_candidates.py`
- `FilterStats`：
  - 欄位：`frames`, `kept`, `dropped`, `area_rejected`, `stability_rejected`, `cascade_rejected`；
  - `add_frame(...)`：更新單一 frame 的統計；
  - `to_dict()`：輸出 JSON 友善格式，用於 manifest 與 log。
- `bbox_from_mask(mask)`：從布林遮罩重建 XYXY bbox，與 SSAM 產生器一致。
- `filter_level(level_root, min_area, stability_threshold, use_cascade, verbose)`：
  - 透過 `RawCandidateArchiveReader` 讀取 raw 候選；
  - 補齊 `computed_area` 與 `stability` 欄位；
  - 若 `use_cascade=True` 且資料具備 `unique_id`，則呼叫 `CascadeFilter` 進行 parent-aware 篩選；
  - 寫出 `filtered/filtered.json`，保存每一 frame 的精簡候選與 bbox／mask。
- `run_filtering(root, levels, ...)`：
  - 逐層呼叫 `filter_level`，彙整統計資訊；
  - 可選擇將濾波設定與統計寫回 `manifest.json`。
- `parse_args()`／`main()`：提供獨立 CLI，允許在不重新跑 Semantic-SAM 的前提下，調整濾波閾值。

---

## Step 2 ─ SAM2 追蹤（Tracking）

### `track_from_candidates.py`

> 追蹤子系統的內部結構（候選載入、DedupStore、sam2_runner 等）詳見 `tracking/TRACKING_GUIDE.md`。此處僅說明入口模組。

- 環境與路徑：
  - 於 `__main__` 分支中，將 `src/` 與 SAM2 專案根加入 `sys.path`；
  - 設定 `TMPDIR` 等暫存目錄到專案根下的 `tmp/`，避免塞滿系統磁區；
  - 透過 `pipeline_defaults` 解析 `DEFAULT_SAM2_ROOT`、`DEFAULT_SAM2_CFG`、`DEFAULT_SAM2_CKPT`。
- 公用函式：
  - `resolve_sam2_config_path(config_arg)`：將 CLI 傳入的絕對 YAML 路徑轉為 SAM2 hydra-style 相對路徑（或保留原樣）。
  - `configure_logging(explicit_level)`：利用 `utils.logging.setup_logging` 設定 log 層級。
- `run_tracking(...)`：
  - 驗證並整理輸入參數（levels、mask_scale_ratio 等）；
  - 透過 `prepare_tracking_context`（tracking.pipeline_context）載入 SSAM manifest／frame 選取資訊等；
  - 設定 `PYTORCH_CUDA_ALLOC_CONF` 以減少 OOM；
  - 切換工作目錄至 SAM2 root，呼叫 `build_sam2_video_predictor` 建立追蹤模型；
  - 建立子影片（subset video）與 frame 對應表；
  - 建構跨層級的 `ProvenanceTracker`；
  - 逐 level 呼叫 `tracking.level_runner.run_level_tracking` 實際執行 SAM2 追蹤與產物寫入；
  - 結束後回傳最終輸出根目錄路徑（含 `video_segments`／`object_segments`／比較視覺化等）。
- `main()`：對應 CLI `python -m my3dis.track_from_candidates`。

---

## Step 3 ─ 家族樹與關聯結構（Relations / Family Tree）

### `relation_index.py`
- `RelationIndexWriter`：
  - `add_object(obj_id, parent_id=None, frames=None)`：登記物件在各幀的出現情況與父節點；
  - `add_frame(frame_idx, frame_name, objects)`：記錄單一 frame 的可見物件集合；
  - `set_source(key, path)`：以相對路徑記錄來源檔案（`video_segments`、`object_segments`、`filtered_candidates`）；
  - `write(output_path=None)`：寫出 `index.json`，包含 `meta`／`objects`／`frames` 三部分。
- `RelationIndexReader`：
  - 封裝 `index.json` 的讀取與查詢；
  - `get_object(obj_id)`／`get_frame(frame_idx)`：回傳個別條目；
  - `get_source_path(key)`：取得來源檔案相對路徑。
- `build_level_index(...)`：
  - 讀取 object/video segments 的 manifest，構建 `RelationIndexWriter`，並生成單層級 index。
- `_extract_from_object_segments`／`_extract_from_video_segments`：
  - 分別從 object-major／frame-major NPZ manifest 中抽取對應資訊。
- `build_cross_level_relations(run_dir, levels)`：
  - 從 `relations/tree_L{L}.json` 或 legacy `relations/tree.json` 讀入各層級樹狀結構；
  - 建立跨層級 parent-child 關聯與 ancestor path；
  - 計算 `descendant_count` 與統計資訊；
  - 產生 v3 `relations.json`，包含 `hierarchy`、`paths` 與 `statistics`。
- `save_ssam_relations(progressive_results, save_root, levels)`：
  - 將 SSAM progressive refinement 的 tree 結果轉換成與 SAM2 一致的 relations 形式，簡化後續 family tree 建構。

### `family_tree_builder.py`
- `load_provenance_tree(tree_path)`：讀取 per-level provenance tree（SSAM or SAM2）。
- `load_video_segments_frames(npz_path)`：
  - 支援多種 NPZ 命名（新 ZIP 格式／舊 scale-based 命名）；
  - 回傳 `object_id → [frame_indices]` 映射。
- `build_family_tree(run_dir, levels, mask_scale_ratio=0.3)`：
  - 優先使用 `unified_provenance_tree.json`；若不存在則回退到 per-level provenance；
  - 合併所有層級物件為單一 registry，紀錄 level、parent、children、virtual_children、frames 以及 `mask_archive` 路徑；
  - 透過 `build_families` 建立連通元件作為 family；
  - 計算統計：各層級物件數、orphan 數量、virtual children 次數等；
  - 回傳 family tree dict，並可選擇寫入 `relations/family_tree.json`。
- `build_families(objects, parent_map)`：以 parent-child 圖的連通元件為單位，為每個 root 建立 family 物件。
- `collect_descendants(root_id, objects)`：遞迴收集某 root 的所有子孫。
- `save_family_tree(tree, output_path)`：將 family tree 以排版良好的 JSON 輸出。
- `main()`：CLI 入口，供對既有 SAM2 run 事後建立 family tree。

### `family_tree_query.py`
- `FamilyTreeQuery`：
  - `__init__(family_tree_path)`：載入 family tree，建立 `objects` 字典與 mask archive 快取。
  - 基本查詢：
    - `get_object_info(obj_id)`／`get_object_frames(obj_id)`；
    - `get_parent(obj_id)`／`get_children(obj_id)`；
    - `get_virtual_children(obj_id)`／`has_virtual_children(obj_id)`／`get_all_children(obj_id, include_virtual)`。
  - 家族結構：
    - `get_siblings(obj_id)`、`get_family_root(obj_id)`、`get_family_members(obj_id)`；
    - `get_common_frames(obj_ids)`：計算多個物件同時出現的幀。
  - 資料載入：
    - `load_mask(obj_id, frame_idx)`：從 video segments NPZ 載入指定物件的某一幀遮罩（自動處理新舊格式與 fallback）。
    - `load_masks_batch(obj_id, frame_indices=None)`：批次載入多幀遮罩。
  - 檢索與輸出：
    - `get_statistics()`：回傳 family tree 中的統計欄位；
    - `search_objects_by_level(level)`／`search_objects_by_frame(frame_idx)`；
    - `export_family_subgraph(obj_id, output_path)`：輸出單一 family 的子圖 JSON，便於視覺化或分析。
- `main()`：提供簡易 CLI，便於互動式檢查 family tree。

### `recover_relations.py`
- `compute_iou(mask1, mask2)`：計算二維遮罩的 IoU。
- `compute_containment(child, parent)`：計算 child 在 parent 中的覆蓋比例。
- `load_object_masks_from_tracking(level_dir, mask_scale_ratio)`：
  - 讀取多種命名規則下的 video_segments 檔案；
  - 回傳 `object_id → frame_idx → mask`。
- `find_parent_candidates(child_id, child_masks, parent_level_objects, containment_threshold)`：
  - 針對每個 child，計算與 parent level 上物件的 containment 比例並排序，產生最佳親代候選列表。
- `recover_legacy_relations(experiment_dir, levels, containment_threshold)`：
  - 在缺乏現代 `tree.json` 的舊實驗上，自動建構 parent-child 關係；
  - 生成與 `family_tree_builder` 相容的 relations 檔案。
- `main()`：CLI 入口，常用於批次遷移舊版本實驗。

### `format_utils.py`
- 格式偵測：
  - `detect_relations_format(data)`：判斷 `relations.json` 格式版本（1.0 / 2.0 / 2.5 / 3.0 / unknown）。
  - `detect_index_format(data)`：判斷 `index.json` 版本。
  - `detect_npz_manifest_format(data)`：判斷 NPZ manifest 版本。
- 轉換與統計：
  - `normalize_relations_v3(data)`：將舊版 relations 結構轉換成 v3 統一格式（含 `hierarchy`／`paths`／`statistics`）。
  - `get_object_count(relations_data)`：無論版本為何，都回傳總物件數。
  - `validate_format_v3(data, data_type)`：檢查 relations/index/npz_manifest 是否符合 v3 期望欄位與 schema。
  - `load_relations_compat(path)`：封裝「讀檔 → 偵測 → 正規化」流程，保證回傳值皆為 v3 格式。

---

## Step 4 ─ 3D 聚合與 SigLIP 特徵指派

### `aggregation/`（3D Mask Aggregation）
- `aggregation_pipeline.py`：
  - `detect_available_levels(run_dir)`：根據 run 目錄中存在的輸出自動推斷可用層級。
  - `_is_family_tree_format`／`_family_tree_to_relations`／`_relations_to_family_tree`／`_prepare_relations_data`：
    - 在 `family_tree.json` 與 legacy `relations.json` 之間進行格式轉換，以保持後續流程一致。
  - `AggregationPipeline.__init__(family_tree_path, pointcloud_path, depth_maps_dir, color_frames_dir, camera_params_path, iou_threshold, min_volume, vis_depth_threshold, device)`：
    - 建立 Mask projector、3D 聚合器與 Search3D 匯出端。
  - `_init_world2cam(camera_params_path)`：讀取相機內外參，建立 world→camera 轉換矩陣。
  - `AggregationPipeline.run(output_dir, levels)`：
    - 逐層由 family tree 載入物件遮罩；
    - 呼叫 `MaskProjector` 進行 2D→3D 投影；
    - 透過 `aggregate_masks_3d`／`filter_proposals_by_volume`／`compute_iou_3d` 聚合與去重；
    - 寫出 `proposal_data_level{L}.npz` 與 `aggregation_metadata.json`。
  - `main()`：對應 `python -m my3dis.run_aggregation` CLI。
- `mask_projection.py`／`mask_projection_3d_to_2d.py`：
  - `MaskProjector`：提供 `project_mask_to_3d` 與深度圖載入／投影的輔助函式；
  - `Mask3Dto2DProjector`：提供 `project_3d_mask_to_2d` 與 `get_visible_frames`，便於從 3D proposal 回投影至 2D 影像。
- `aggregation_utils.py`：
  - `aggregate_masks_3d`：依 union/intersection/majority 等方法整合跨幀遮罩；
  - `compute_proposal_geometry`：計算每個 proposal 的幾何性質（質心、bbox、體積等）；
  - `filter_proposals_by_volume`：根據體積閾值過濾；
  - `compute_iou_3d`：計算 3D proposal 之間的 IoU。
- `proposal_loader.py`：
  - `ProposalLoader`：從 experiment 目錄載入指定層級的 proposal NPZ；
  - `_find_level_dir`／`_find_tracking_files`／`_aggregate_masks`：輔助解析路徑與整併遮罩；
  - `load_proposals_from_experiment(experiment_root, levels)`：高階 API，供評估腳本使用。
- `search3d_formatter.py`：
  - `export_predictions_search3d`／`export_predictions_both_formats`：將預測結果匯出為 Search3D 或自訂格式；
  - `load_point_cloud_for_export`：根據 proposal metadata 載入點雲；
  - `validate_search3d_format`：檢查輸出 payload 是否符合 Search3D 的 shape/dtype 要求。

### `siglip_assignment/`（SigLIP Feature Assignment）
- `siglip_pipeline.py`：
  - `SigLIPAssignmentPipeline.__init__(proposal_dir, color_frames_dir, depth_maps_dir, model_name, batch_size, topk_frames, device)`：
    - 建立特徵擷取模組與 proposal 載入器。
  - `run(output_dir, levels, object_labels=None)`：
    - 逐層載入 proposals；
    - 透過 `_extract_features_for_proposals` 取得 SigLIP 特徵；
    - 若提供 `object_labels`，再呼叫 `_assign_labels` 進行開放詞彙分類；
    - `_export_results` 與 `_export_metadata` 寫出 `siglip_features_level{L}.npz` 及 JSON metadata。
  - `main()`：對應 `python -m my3dis.run_siglip_assignment` 的入口。
- `feature_extraction.py`：
  - `SigLIPFeatureExtractor`：
    - `__init__(model_name, device, cache_dir)`：載入 SigLIP 模型與 Processor；
    - `extract_batch(frames, crops, mask_metadata)`：對多幀、多比例 crop 進行 batched 推論並平均特徵；
    - `_extract_from_frame`／`_extract_single_proposal`：將 proposal bbox 投影為實際影像 crop；
    - `get_text_features(labels)`：將文字標籤轉為嵌入向量。
- `assignment_utils.py`：
  - `aggregate_features(frame_features, weights)`：對多視角特徵進行加權平均；
  - `assign_labels(features, text_embeddings, temperature, allow_duplicates)`：套用 cosine similarity + 溫度縮放，產生類別與分數；
  - `assign_part_labels(object_features, part_features, temperature)`：在物件／部件特徵上融合訊息；
  - `compute_mask_scores(proposal_masks, per_point_scores)`：由點級分數回推到 proposal 級別。

---

## Step 5 ─ 報表與工作流程（Reporting & Workflow）

### `generate_report.py`
- `StageTiming` dataclass：包裝每個 stage 的耗時與 GPU 資訊，提供 `duration_text` 屬性。
- 輔助函式：
  - `load_json(path)`：溫和讀取 JSON（失敗時回傳 `None`），避免破壞報告生成。
  - `collect_stage_timings(summary)`：由 workflow summary 中萃取各 stage 的時間資訊並排序。
  - `pick_first_mid_last(items)`：在多張比較圖中挑選「第一／中位／最後」三張代表圖，避免報告過於冗長。
  - `downscale_image(src, dst, max_width)`：縮小圖像並寫入 report 資料夾。
  - `render_level_section(level, viz_dir, report_dir, ...)`：為單一 level 產生 Markdown 表格與貼圖連結。
- `build_report(run_dir, report_name='report.md', max_preview_width=960)`：
  - 彙整 manifest、workflow summary 與比較視覺化；
  - 產生包含實驗設定、耗時表、各層級代表圖的 Markdown 報告。
- `parse_args()`／`main()`：提供 CLI，在完成的 run 目錄上產生報告。

### `workflow/`（多場景 YAML 工作流程）

> 詳細說明請見 `workflow/WORKFLOW_GUIDE.md`。此處僅列出與外部互動較密切的模組。

- `run_workflow.py`：
  - `_configure_entry_logging()`：配置全域 logging 格式（含 PID 與 timestamp）。
  - `_stdout_descriptor()`：檢查目前 stdout 的實際指向（檔案、管線或 socket）。
  - `_record_pid_map(config_path)`：將每次執行的 PID、命令列與設定檔紀錄到 `logs/run_pid_map.csv`。
  - `main()`：解析 `--config`、`--dry-run` 等參數，載入 YAML，並呼叫 `workflow.execute_workflow` 執行所有場景。
- `workflow/core.py`：
  - 作為相容層，匯出 `SceneContext`, `SceneWorkflow`, `run_scene_workflow`, `execute_workflow`。
- `workflow/scene_workflow.py`：
  - `SceneContext`：封裝 YAML 設定、路徑與場景級 metadata。
  - `SceneWorkflow`：
    - `run()`：依序執行 `ssam`／`filter`／`tracker`／`report` 等 stage；
    - `_run_ssam_stage`／`_run_filter_stage`／`_run_tracker_stage`／`_run_report_stage`：調用對應 CLI 模組（stage runner），並記錄 StageRecorder。
    - `_finalize()`：寫出 `workflow_summary.json`、`environment_snapshot.json`，並呼叫 `apply_scene_level_layout`／`append_run_history`。
- `workflow/executor.py`：
  - 提供 `execute_workflow(config, override_output, config_path, memory_event_paths)` 作為 YAML 模式的總入口；
  - 支援多進程並行執行多個場景，並與 `oom_monitor` 整合以偵測記憶體不足事件。
- `workflow/summary.py`：
  - `StageRecorder`：以 context manager 方式統一記錄各 stage 的開始／結束時間、資源使用與例外狀態；
  - `collect_environment_snapshot`、`export_stage_timings`、`apply_scene_level_layout`、`append_run_history` 等函式，則負責產出 summary JSON、調整輸出目錄結構與維護 `logs/workflow_history.csv`。
- `workflow/stage_config.py`：
  - `SSAMStageConfig`／`TrackingStageConfig`／`FilterStageConfig`：
    - `from_yaml_config(...)` 會將 YAML 中的 `stages.ssam`／`stages.tracker`／`stages.filter` 段落轉為型別安全的 dataclass，並集中處理所有驗證邏輯；
    - `to_legacy_kwargs()` 則用於與舊式函式介面（例如 `run_generation`、`run_filtering`）銜接。
- `workflow/stage_runner.py` 與 `workflow/stages/`：
  - `StageRunner` 為抽象基底類別，定義 `name`／`should_run()`／`execute()` 介面與 `_stage_cfg`／`_stage_summary` 等共通輔助；
  - `stages/ssam_stage.py`／`filter_stage.py`／`tracker_stage.py`／`family_tree_stage.py`／`family_viz_stage.py`／`report_stage.py` 等，則各自實作對應 stage 的實際邏輯，並透過 `StageRecorder` 回寫 summary。

### `prepare_tracking_run.py`
- `build_destination(source, dest_root, name)`：決定複製 run 目錄的目的路徑（可以依照時間戳自動命名）。
- `clone_directory(source, dest, mode)`：
  - `mode='link'`：盡可能使用硬連結，節省空間；
  - `mode='copy'`：進行深度複製。
- `update_config(config_path, new_run_dir)`：更新 YAML 中 `experiment.run_dir` 欄位，讓後續 SAM2 追蹤指向新 run 目錄。
- `parse_args()`／`main()`：提供 CLI 以便快速複製既有實驗並調整配置。

---

若需要針對特定子系統深入了解，建議搭配以下文件閱讀：

- 追蹤內部實作：`src/my3dis/tracking/TRACKING_GUIDE.md`
- YAML 工作流程：`src/my3dis/workflow/WORKFLOW_GUIDE.md`
- 3D proposal 聚合：`src/my3dis/aggregation/README.md`
- SigLIP 特徵指派：`src/my3dis/siglip_assignment/README.md`
- 開放詞彙推論：`src/my3dis/inference/OVERVIEW.md`

