# Tracking 子模組導覽（Step 2）

> `my3dis.tracking` 負責在 Semantic-SAM 濾波之後，載入 2D 候選遮罩、準備 SAM2 提示、執行影片追蹤，並將結果打包成後續 family tree、3D 聚合與推論所需的標準化檔案。本文件依實際執行順序介紹各模組、類別與主要函式。

## 每層級追蹤流程概觀

```text
run_level_tracking(level, ...) ─────────────────────────────────────────────┐
  ├─ load_filtered_manifest(level_root)                                    │
  ├─ frames_meta = filtered.json['frames']                                 │
  ├─ preview_local_indices = select_preview_indices(...)                   │
  ├─ candidate_iter = iter_candidate_batches(level_root, frames_meta, ...) │
  ├─ dedup_store = DedupStore()                                            │
  ├─ frame_store = FrameResultStore()                                      │
  └─ sam2_tracking(subset_dir, predictor, candidate_iter, ...)             │
       ├─ predictor.init_state(video_path=subset_dir)                      │
       ├─ for each FrameCandidateBatch:                                    │
       │    ├─ prepared = _prepare_prompt_candidates(batch.candidates)     │
       │    ├─ filtered = _filter_new_candidates(prepared, dedup_store)    │
       │    ├─ _add_prompts_to_predictor(predictor, state, filtered, ...)  │
       │    └─ frame_segments = _propagate_frame_predictions(...)          │
       │         └─ result: {abs_idx: {obj_id: packed_mask}}               │
       │              ├─ dedup_store.add_packed(abs_idx, ...)              │
       │              └─ frame_store.update(abs_idx, frame_name, ...)      │
       └─ return TrackingArtifacts(object_refs, preview_segments, ...)     │

  ├─ artifacts = persist_level_outputs(..., frame_store, ...)              │
  │    ├─ build_video_segments_archive(iter_frames(), video_segments.npz)  │
  │    └─ build_object_segments_archive(object_refs, object_segments.npz)  │
  ├─ if render_viz:                                                        │
  │    └─ save_comparison_proposals(viz_dir, base_frames_dir, ...)         │
  └─ return LevelRunResult(artifacts, comparison, warnings, stats, timer)  │
```

---

## Step 2‑0 ─ 匯出與共用輔助（`__init__.py`、`helpers.py`）

### `tracking/__init__.py`
- 重新匯出常用型別與工具，方便 `track_from_candidates.py` 等入口模組一次性 import：
  - 計時工具：`TimingAggregator`, `format_duration_precise`
  - 幾何／縮放工具：`format_scale_suffix`, `scaled_npz_path`, `resize_mask_to_shape`, `infer_relative_scale`, `determine_mask_shape`, `compute_iou`
  - 追蹤結果型別：`TrackingArtifacts`, `LevelRunResult` 等（由其他子模組定義）

### `helpers.py`
- `ProgressPrinter`：簡易進度列，用於 CLI log。
  - `__init__(total)`：設定總 frame 數。
  - `update(index, abs_frame)`：更新目前進度與絕對 frame index，輸出單行文字（可被覆寫）。
  - `close()`：在結束時計一個換行，避免後續 log 黏在同一行。
- `_TimingSection`（內部類別）：
  - 供 `TimingAggregator.track(stage)` 當作 context manager 使用，記錄某一段程式區塊的耗時。
- `TimingAggregator`：
  - `add(stage, duration)`：累計某個標籤的時間。
  - `track(stage)`：回傳 `_TimingSection` 物件，搭配 `with` 自動紀錄時間。
  - `total(stage)`／`total_prefix(prefix)`／`total_all()`：查詢單一 stage、特定前綴或全部的時間累計。
  - `items()`：依插入順序回傳 `(stage, duration)` 列表，用於 log 或 summary。
  - `merge(other)`：將另一個 aggregator 的結果合併進來。
  - `format_breakdown()`：產生人類可讀的時間分解摘要字串。
- 幾何與縮放：
  - `format_scale_suffix(ratio)`：將縮放比例轉成檔名友善的後綴（例如 `_scale0.3x`）。
  - `scaled_npz_path(path, ratio)`：根據比例調整 NPZ 檔名。
  - `resize_mask_to_shape(mask, target_shape)`：以最近鄰插值將遮罩調整到指定尺寸，用於 IoU 計算時對齊解析度。
  - `infer_relative_scale(mask_entry)`：從 packed mask payload 中推斷已套用的縮放比例。
  - `determine_mask_shape(mask_entry, fallback)`：取得原始遮罩的高寬。
  - `compute_iou(mask1, mask2)`：計算兩個遮罩的 IoU，必要時進行升／降採樣。
- 時間：
  - `format_duration_precise(seconds)`：以較高精度格式化秒數，用於追蹤 log。

---

## Step 2‑1 ─ 載入 Semantic‑SAM 候選（`candidate_loader.py`）

### `candidate_loader.py`
- `FrameCandidateBatch`（dataclass）：
  - 欄位包含：
    - `local_index`／`absolute_index`：局部與全域 frame index；
    - `frame_name`：檔名；
    - `candidates`：單幀所有候選遮罩的列表（已含 bbox、area 等）。
- `load_filtered_manifest(level_root)`：
  - 讀取 `level_{L}/filtered/filtered.json`，回傳 frames metadata；
  - 若不存在，會回報合理錯誤訊息供 CLI 使用者排查。
- `_load_frame_candidates(level_root, frame_meta, mask_scale_ratio, ...)`：
  - 從 raw/filtered 檔案載入某一幀的候選；
  - 視需要還原／縮放遮罩，並整理成統一格式（含 packed mask 與 bbox）。
- `iter_candidate_batches(level_root, frames_meta, mask_scale_ratio, ...)`：
  - 逐幀產生 `FrameCandidateBatch`；
  - 內部會呼叫 `_load_frame_candidates`，並將 frame index、名稱與候選封裝在 dataclass 中。
- `load_filtered_frame_by_index(level_root, local_idx, mask_scale_ratio)`：
  - 以「局部 index」載入單一幀的候選，主要供比較視覺化抽樣使用。
- `_build_preview_segment(...)`／`_build_preview_stub(...)`：
  - 建立體積較小的預覽資料結構，只保留必須欄位（面積、分數、bbox）；視覺化模組透過這些 stub 建構比較圖。

---

## Step 2‑2 ─ 去重與中繼儲存（`stores.py`）

### `stores.py`
- `_frame_entry_name(frame_idx)`：
  - 生成統一的 frame entry 名稱，用於 NPZ／ZIP manifest 中，例如 `frame_000123`。
- `DedupStore`：
  - 追蹤已輸出的遮罩或 box，以避免對同一物件重複建立 proposal。
  - `add_packed(abs_idx, obj_id, mask_entry)`：註冊一個已輸出的 packed mask。
  - `seen_packed(mask_entry, iou_threshold)`：
    - 與既有 packed mask 計算 IoU；
    - 若高於 `iou_threshold`，視為重複，回傳 True。
  - `seen(box, mask_entry, iou_threshold)`：
    - 當無法直接比對 raw mask 時，使用 bounding box + mask entry 的折衷策略進行比對。
- `FrameResultStore`：
  - 收集追蹤過程中每個 frame 的輸出與預覽資料。
  - `update(abs_idx, frame_name, packed_masks, preview)`：
    - 將 SAM2 對某幀的輸出（packed masks）與預覽資料寫入 internal 結構；
  - `iter_frames()`：
    - 以排序好的順序產生 frame-level 資料，用於 `build_video_segments_archive`。
  - `iter_preview_segments()`：
    - 產生抽樣預覽的資料（frame index、bbox、mask 等），供視覺化模組使用。
  - `_iter_sorted(items)`：
    - 內部輔助，用於確保輸出順序 deterministic。
- `TrackingArtifacts`（dataclass）：
  - 封裝 SAM2 追蹤完一個 level 後的主要產物：
    - object references、preview segments、dedup 統計與 timing 資訊；
  - 後續 `level_runner` 與 `outputs` 皆以此為單位操作。

---

## Step 2‑3 ─ 追蹤 Context 與 Manifest 更新（`pipeline_context.py`）

### `pipeline_context.py`
- `PipelineContext`：
  - 承載單層級追蹤所需的所有資訊，包括：
    - SSAM manifest、run 目錄、
    - level 列表、選取幀索引、SSAM 頻率、
    - subset video 位置等。
- `build_pipeline_context(...)`／（新版命名可能為 `prepare_tracking_context`）：
  - 由 `track_from_candidates.run_tracking` 呼叫；
  - 讀取 `manifest.json` 與 `filtered/filtered.json`；
  - 解析 levels、frame range 與 SSAM 頻率；
  - 決定比較視覺化所需的抽樣幀。
- `select_preview_indices(frames_meta, sample_stride, max_samples)`：
  - 按照 stride 或最大樣本數，從全部 frames 中選擇用於比較視覺化的幀索引；
  - 保證第一／最後幀具有較高機率被選中。
- `update_tracking_manifest(manifest, tracking_stats, level, artifacts)`：
  - 將追蹤統計結果寫回 manifest：
    - 每層級的追蹤幀數、dedup 統計、輸出路徑等。
- `summarise_tracking_stats(level_results)`：
  - 將所有 level 的 `LevelRunResult` 彙整成一份摘要，供 workflow summary 使用。

---

## Step 2‑4 ─ SAM2 追蹤核心（`sam2_runner.py`）

### `sam2_runner.py`
- `_prepare_prompt_candidates(candidates, mask_scale_ratio, long_tail_box_prompt, all_box_prompt)`：
  - 將 SSAM 候選轉換為 SAM2 接受的提示（prompt）：
    - 預設以 mask 為主；若必要則退而求其次改用 bbox；
    - 根據 prompt mode（全遮罩、長尾 bbox、全框）做對應設定。
- `_filter_new_candidates(prepared, dedup_store, iou_threshold)`：
  - 比對新的候選與 DedupStore 中已存在的遮罩；
  - 對於 IoU 過高的候選直接略過，避免重複追蹤；
  - 同時記錄必要的統計資訊。
- `_add_prompts_to_predictor(predictor, state, candidates, prompt_mode)`：
  - 依設定將 prompt 加入 SAM2 predictor：
    - 控制 masks／boxes 的權重；
    - 可能根據 `prompt_mode` 決定是否對長尾物件加強提示。
- `_propagate_frame_predictions(predictor, state, start_frame, end_frame, ...)`：
  - 以指定的 propagating 策略（單向／雙向、步長等）在影片上傳播遮罩；
  - 收集每一幀的 packed mask 結果。
- `sam2_tracking(subset_dir, predictor, candidate_iter, iou_threshold, ...)`：
  - 初始化 SAM2 狀態（`predictor.init_state`）；
  - 迭代 `FrameCandidateBatch`，對每批候選進行：
    1. 準備 prompts；
    2. 去重；
    3. 加入 predictor 並傳播；
    4. 更新 `FrameResultStore` 與 `DedupStore`；
  - 回傳 `TrackingArtifacts`，包含：
    - object references（供 `object_segments` 建立）、
    - preview segments（供比較視覺化）、以及 dedup 統計。
- `_build_tracking_result(...)`：
  - 將 dedup 統計、FrameResultStore 與抽樣預覽整合成最終 `TrackingArtifacts`。

---

## Step 2‑5 ─ 層級執行與輸出封裝（`level_runner.py`）

### `level_runner.py`
- `LevelRunResult`（dataclass）：
  - 封裝單一 level 追蹤完成後的：
    - `artifacts`（`TrackingArtifacts`）、
    - `comparison`（比較視覺化相關資料）、
    - `warnings`（執行中產生的警告）、`stats`（統計）、`timer`（`TimingAggregator`）。
- `run_level_tracking(level, context, predictor, subset_dir, ...)`：
  - 整體 orchestrator：
    1. 透過 `load_filtered_manifest` 與 context 建立 `frames_meta`；
    2. 建立 `DedupStore` 與 `FrameResultStore`；
    3. 呼叫 `sam2_tracking` 執行實際追蹤；
    4. 將結果交給 `_persist_level_outputs` 寫入檔案；
    5. 返回 `LevelRunResult`。
- `_build_level_stats(level, artifacts, timer)`：
  - 根據 `TimingAggregator` 與 dedup 統計建構可讀的統計摘要（總物件數、有效幀數、追蹤耗時等）。
- `_persist_level_outputs(level_root, artifacts, mask_scale_ratio, comparison_cfg)`：
  - 呼叫 `outputs.build_video_segments_archive`：
    - 以 frame-major 方式將遮罩寫入 `video_segments_L{L}.npz`；
  - 呼叫 `outputs.build_object_segments_archive`：
    - 以 object-major 方式建立 `object_segments_L{L}.npz`；
  - 更新 manifest 中對應層級的輸出路徑與統計。
- `_render_comparison(...)`：
  - 若啟用 `render_viz`，呼叫 `outputs.save_comparison_proposals` 產生比較圖（原始 frame vs SSAM vs SAM2）。

---

## Step 2‑6 ─ 輸出編碼與比較視覺化（`outputs.py`）

### `outputs.py`
- JSON 編碼：
  - `encode_packed_mask_for_json`／`decode_packed_mask_from_json`：
    - 在 JSON 中安全儲存／還原 packed mask（bitstream + shape）的序列化形式。
- 路徑與取樣：
  - `_ensure_frames_dir(path)`：確保「frames 子目錄」存在。
  - `_frame_entry_name(frame_idx)`：使用與 `stores` 相同的命名規則建立 entry 名稱。
  - `_normalize_stride(value)`／`_normalize_max_samples(value)`：解析 YAML／CLI 中的抽樣參數。
  - `_downsample_evenly(values, target)`：
    - 在不破壞第一／最後元素的前提下，平均下採樣序列；
  - `_apply_sampling_to_frames(frames, sample_stride, max_samples)`：
    - 將 stride 與最大樣本數合併，產生最終需渲染的 preview frame 列表。
- 主要輸出：
  - `build_video_segments_archive(frames, path, mask_scale_ratio, metadata)`：
    - 由 `FrameResultStore.iter_frames()` 提供 frame-major 資料；
    - 寫出 ZIP/NPZ 檔與 `manifest.json`，包含：
      - 每幀的 frame index/name；
      - 每個物件在該幀的 packed mask；
      - 以及 mask 縮放資訊。
  - `build_object_segments_archive(object_manifest, path, mask_scale_ratio, metadata)`：
    - 以 object-major 視角建立對應 ZIP/NPZ；
    - manifest 中紀錄每個物件於哪些幀可見。
  - `save_comparison_proposals(viz_dir, base_frames_dir, subsets, ...)`：
    - 根據 sampling 結果選擇一組代表 frame；
    - 從原始影像或 subset video 載入底圖；
    - 疊加 SSAM 與 SAM2 遮罩／bbox；
    - 將 PNG 與 JSON metadata 輸出於 `viz/compare/` 之下，並在遇到 fallback 或缺檔時寫入警告（供 workflow summary 使用）。

---

## Step 2‑7 ─ 相容介面與譜系追蹤（`compat.py`、`provenance_tracker.py`）

### `compat.py`（Legacy 輸出相容層）
- `LegacyFrame`：
  - dataclass，代表舊版框架使用的 frame-major 結構：
    - `frame_index`、`frame_name`、`objects: Dict[int, Dict[str, Any]]`。
- `_read_json_from_zip(zf, entry)`：
  - 從 ZIP 檔中讀取指定 JSON entry，並轉為 dict。
- `_decode_object_payload(payload, unpack_masks)`：
  - 先透過 `decode_packed_mask_from_json` 還原 packed entry；
  - 若 `unpack_masks=True`，再呼叫 `unpack_binary_mask` 轉為布林遮罩並塞回 payload。
- `iter_legacy_frames(archive_path, unpack_masks=False, frame_filter=None)`：
  - 以 generator 方式遍歷 `video_segments` ZIP；
  - 對符合 `frame_filter` 的 frame 產生 `LegacyFrame` 物件；
  - 提供近似舊版 `np.savez` 輸出的迭代介面。
- `load_legacy_frame_dict(archive_path, unpack_masks=False)`：
  - 回傳 `{frame_idx: {obj_id: payload}}` 的 dict，與舊版程式碼期待的結構相容。
- `load_legacy_video_segments(archive_path, unpack_masks=False)`：
  - 回傳 `(frames_dict, manifest)` 的 pair，模擬舊版 npz 輸出。
- `load_object_segments_manifest(object_archive_path)`：
  - 讀取 `object_segments` manifest 中的 object-major mapping；
  - 回傳 `{obj_id: {frame_idx: {'frame_entry': str}}}` 結構，供 legacy 工具查詢。

### `provenance_tracker.py`（譜系追蹤）
- `ProvenanceTracker`：
  - 設計目標是以 **統一映射** 的方式追蹤 SSAM `unique_id` 與 SAM2 物件 id 之間的對應關係，無論該 SSAM mask 最後是「被接受」還是「因 dedup 而被拒絕」。
  - 內部主要儲存：
    - `ssam_to_sam2: Dict[str, int]`：SSAM id → SAM2 物件 id；
    - `sam2_provenance: Dict[int, Dict[str, Any]]`：SAM2 物件對應 SSAM 父節點、層級、frame index、lineage 等；
    - `unresolved_rejections: List[tuple]`：紀錄少數無法解析的 dedup 情況；
    - `virtual_children: Dict[int, List[str]]`：SAM2 parent 物件 id → 多個「被 dedup 合併」的子 SSAM id。
- 註冊方法：
  - `register_accepted_prompt(sam2_obj_id, ssam_unique_id, ssam_parent_id, ssam_frame_idx, level, lineage=None)`：
    - 將一個被 SAM2 接受為新物件的 SSAM mask 登記到雙向 mapping 中；
    - 保存其父節點 id、層級、frame index 與完整 lineage。
  - `register_rejected_prompt(ssam_unique_id, matched_sam2_obj_id, ssam_parent_id=None)`：
    - 將因 dedup 而被拒絕的新 SSAM mask 登記為「對某個既有 SAM2 物件的別名」；
    - 若 `ssam_parent_id` 存在，則會找出父節點對應的 SAM2 物件 id，並將該 SSAM id 註記為該 parent 的 virtual child。
  - `register_unresolved_rejection(ssam_unique_id, matched_mask_idx, frame_idx)`：
    - 紀錄無法透過 mapping 完整解析的拒絕事件，方便後續偵錯 orphan 問題。
- 查詢方法：
  - `find_parent(sam2_obj_id)`：
    - 根據 `sam2_provenance` 查出該 SAM2 物件對應的 SSAM parent id，再透過 `ssam_to_sam2` 找到 SAM2 parent 物件 id；
    - 時間複雜度為 O(1)。
  - `get_virtual_children(sam2_obj_id)`／`has_virtual_children(sam2_obj_id)`：
    - 查詢某個 SAM2 物件是否具有 virtual children，以及其 SSAM id 列表。
  - `get_family_hierarchy()`：
    - 輸出 child→parent 的關係映射，並附帶孤兒統計；
    - `family_tree_builder` 可藉此更準確重建跨層級家族樹，而不必依賴事後幾何包含度分析。

---

整體而言，`my3dis.tracking` 將 Semantic‑SAM 的 2D 候選與 SAM2 的影片追蹤以明確的資料結構串接起來，並透過 provenance 與 relations 介面為後續的 family tree、3D proposal 與開放詞彙推論提供穩定的基礎。若需要完整工作流程脈絡，請搭配 `src/my3dis/OVERVIEW.md` 與 `workflow/WORKFLOW_GUIDE.md` 一併閱讀。

