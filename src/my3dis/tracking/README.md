# Tracking 子模組說明（對應 Step 2）

> 本目錄涵蓋「SAM2 追蹤」階段的所有協作元件：載入 SSAM 候選、準備 prompt、呼叫 SAM2、儲存結果與視覺化。內容依實際執行順序排列，逐一說明檔案與函式。

## 追蹤階段流程圖（Level 內部）

```
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
  └─ 回傳 LevelRunResult(artifacts, comparison, warnings, stats, timer)     │
```

## Step 2-0：匯出與共用工具

### `__init__.py`
- 將常用工具（時間統計、BBox 轉換、輸出封裝等）集中匯出，讓 `track_from_candidates.py` 可以一次導入。

### `helpers.py`
- `ProgressPrinter`
  - `__init__(total)`：初始化簡易進度列。
  - `update(index, abs_frame)`：更新顯示（包含目前處理到第幾幀）。
  - `close()`：收尾換行，避免覆蓋下一行輸出。
- `_TimingSection`：`TimingAggregator.track()` 專用 context manager，量測區間耗時。
  - `__enter__()` / `__exit__()`：在離開時把耗時累加到對應 stage。
- `TimingAggregator`
  - `__init__()`：建立記錄容器。
  - `add(stage, duration)`：累加指定 stage 的時間。
  - `track(stage)`：回傳 `_TimingSection` 供 `with` 使用。
  - `total(stage)` / `total_prefix(prefix)` / `total_all()`：查詢總耗時。
  - `items()`：依原始加入順序回傳 `(stage, duration)` 列表。
  - `merge(other)`：合併另一個 aggregator 的結果。
  - `format_breakdown()`：輸出人類可讀的 summary 字串。
- `format_scale_suffix(ratio)`：把比例轉為短字串（例如 `0.3` → `0.3`），用於檔案命名。
- `scaled_npz_path(path, ratio)`：依 mask 比例調整輸出檔名（加入 `_scale{ratio}x` 後綴）。
- `resize_mask_to_shape(mask, target_shape)`：把遮罩 resize 成指定大小（最近鄰），確保 IoU 計算一致。
- `infer_relative_scale(mask_entry)`：從 packed mask 中推回降採樣比例。
- `determine_mask_shape(mask_entry, fallback)`：取得遮罩原始尺寸，對 SAM2 輸出與比較圖至關重要。
- `format_duration_precise(seconds)`：以分鐘/小時顯示精確耗時（管控 log）。
- `bbox_transform_xywh_to_xyxy(bboxes)`：把一批 XYWH 轉 XYXY，SAM2 box prompt 需要。
- `bbox_scalar_fit(bboxes, scalar_x, scalar_y)`：依 SAM2 影像大小縮放 bbox。
- `compute_iou(mask1, mask2)`：計算遮罩 IoU，若尺寸不同會進行插值，支援 dedup 與品質檢查。

## Step 2-1：載入 SSAM 候選

### `candidate_loader.py`
- `FrameCandidateBatch`：封裝單幀候選（包含 local/絕對 index 與候選列表）。
- `load_filtered_manifest(level_root)`：讀取 `filtered/filtered.json`。
- `_load_frame_candidates(filt_dir, frame_meta, mask_scale_ratio)`：載入單幀候選並還原遮罩（或套用指定縮放），提供 `iter_candidate_batches` 與預覽功能使用。
- `iter_candidate_batches(level_root, frames_meta, mask_scale_ratio)`：逐幀產生 `FrameCandidateBatch`，供 SAM2 主迴圈迭代。
- `load_filtered_frame_by_index(level_root, frames_meta, local_index, mask_scale_ratio)`：用於視覺化抽樣，按 local index 取出候選。
- `select_preview_indices(total_frames, stride, max_samples)`：挑選代表幀索引，讓比較圖不至於爆量。

## Step 2-2：Dedup 與輸出暫存

### `stores.py`
- `_frame_entry_name(frame_idx)`：統一 frame JSON 檔名格式。
- `_DedupEntry`：儲存單幀的降採樣遮罩堆疊與目標尺寸。
- `DedupStore`
  - `__init__(max_dim)`：設定降採樣最大邊長。
  - `_compute_target_shape(shape)`：根據原始尺寸計算降採樣後的目標形狀。
  - `_ensure_entry(frame_idx, mask_shape)`：取得或初始化 `_DedupEntry`。
  - `_resize(mask, target_shape)`：確保遮罩符合目標尺寸。
  - `_max_iou(entry, candidate)`：計算候選與已加入遮罩的最高 IoU。
  - `has_overlap(frame_idx, mask, threshold)`：判斷候選是否與現有遮罩重疊過多。
  - `add_mask(frame_idx, mask)` / `add_packed(frame_idx, payloads)`：把遮罩加入 dedup 緩衝。
  - `filter_candidates(frame_idx, candidates, threshold)`：過濾掉高重疊的候選，回傳保留清單。
- `FrameResultStore`
  - `__init__(prefix)`：建立暫存資料夾與索引。
  - `update(frame_idx, frame_name, frame_data)`：將 SAM2 預測結果寫入 JSON（以物件 ID 對應 packed mask）。
  - `iter_frames()`：依幀索引順序迭代暫存結果，供打包 `npz`。
  - `cleanup()`：刪除暫存資料夾，避免殘留。

## Step 2-3：追蹤上下文與 manifest 更新

### `pipeline_context.py`
- `TrackingContext`：封裝 manifest、抽樣影格、SSAM 頻率、subset 影片路徑等上下文。
- `LevelRunResult`：描述每個層級追蹤完的輸出（檔案路徑、比較結果、警告、統計與計時器）。
- `prepare_tracking_context(candidates_root, level_list, sam2_max_propagate)`：讀取 manifest、整理 SSAM 序列、決定最終 propagation 參數。
- `resolve_long_tail_area_threshold(manifest, long_tail_box_prompt, all_box_prompt)`：計算小物件改用 box prompt 的面積門檻（支援環境變數覆蓋）。
- `ensure_subset_video(context, data_path, out_root)`：確認 manifest 中的 subset 影格存在，若缺失則重建。
- `update_manifest(context, out_root, level_results, mask_scale_ratio, render_viz)`：把 SAM2 輸出路徑、比較摘要、警告、mask 比例等資訊寫回 manifest。

## Step 2-4：SAM2 追蹤主體

### `sam2_runner.py`
- `PromptCandidate`：裝載保留的 SSAM 候選，用於決定要給 SAM2 的 prompt。
- `TrackingArtifacts`：追蹤階段最終回傳的物件→影格對應、預覽遮罩等資料。
- `_coerce_mask_bool(mask)`：把各種遮罩形式轉為布林陣列。
- `_prepare_prompt_candidates(frame_masks)`：將 `FrameCandidateBatch` 內的項目轉為 `PromptCandidate`，備妥面積與 bbox。
- `_filter_new_candidates(candidates, frame_idx, dedup_store, iou_threshold)`：以 `DedupStore` 過濾重複遮罩。
- `_should_use_box_prompt(candidate, use_box_for_all, use_box_for_small, small_object_area_threshold)`：判斷某個候選要用 box prompt 還是原本的 mask。
- `_add_prompts_to_predictor(predictor, state, frame_idx, candidates, ...)`：實際把 mask/box prompt 加入 SAM2 predictor，回傳下一個可用物件 ID。
- `_propagate_frame_predictions(predictor, state, frame_idx, local_to_abs, total_frames, max_propagate, mask_scale_ratio)`：執行正反向 propagation，把結果轉為 packed mask。
- `sam2_tracking(frames_dir, predictor, candidate_batches, ...)`：Step 2 核心迴圈，遍歷 SSAM frame → 準備 prompt → 呼叫 SAM2 → 寫入 `FrameResultStore`、`DedupStore`、`TrackingArtifacts`。

## Step 2-5：層級執行與輸出打包

### `level_runner.py`
- `persist_level_outputs(level, tracking_output, frame_store, track_dir, mask_scale_ratio, level_timer)`：把暫存 JSON 轉為
  - `video_segments.npz`（frame-major）
  - `object_segments.npz`（object-major）
  並移除暫存檔。
- `run_level_tracking(level, candidates_root, data_path, subset_dir, subset_map, predictor, ...)`：串起整個層級流程：載入候選 → 建立 dedup/frame store → 呼叫 `sam2_tracking` → 儲存輸出 → 抽樣視覺化 → 收集統計，最後回傳 `LevelRunResult`。

## Step 2-6：輸出封裝與比較視覺化

### `outputs.py`
- `encode_packed_mask_for_json(payload)` / `decode_packed_mask_from_json(payload)`：在 JSON 與 numpy packed 格式之間轉換遮罩資料。
- `_ensure_frames_dir(path)`：確保影片 `npz` 檔案所在資料夾存在。
- `_frame_entry_name(frame_idx)`：統一 frame entry 名稱（與 `stores._frame_entry_name` 相同規則）。
- `_normalize_stride(value)`、`_normalize_max_samples(value)`：解析比較圖的採樣步頻與最大輸出張數。
- `_downsample_evenly(values, target)`：在保留頭尾的前提下平均抽樣索引。
- `_apply_sampling_to_frames(frames, sample_stride, max_samples)`：綜合步頻與最大數量條件，決定最終要渲染的幀清單。
- `build_video_segments_archive(frames, path, mask_scale_ratio, metadata)`：把 `FrameResultStore` 產物寫成 ZIP/NPZ，記錄每幀對應的物件清單。
- `build_object_segments_archive(object_manifest, path, mask_scale_ratio, metadata)`：建立物件索引檔（物件 → 對應幀列表）。
- `save_comparison_proposals(viz_dir, base_frames_dir, filtered_per_frame, video_segments, level, ...)`：生成 SAM2 與 SSAM 的比對圖：
  - 決定要渲染哪些幀。
  - 找出原始影像/子集影像。
  - 描繪遮罩與邊框。
  - 儲存 PNG 與摘要 JSON，供 `apply_scene_level_layout` 與報告使用。

---

整體流程建議從 `track_from_candidates.py → pipeline_context → level_runner → sam2_runner → outputs` 順向閱讀，以掌握每個函式在 Step 2 的串接位置。  
若需回顧全流程或其他階段，請參考 `src/my3dis/README.md` 與 `workflow/README.md`。
