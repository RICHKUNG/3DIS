# `mask/` 模組

## `encoding.py`
- 常數 `PACKED_MASK_KEY`, `PACKED_MASK_B64_KEY`, `PACKED_SHAPE_KEY`, `PACKED_ORIG_SHAPE_KEY`：標準化 JSON/NPZ 中記錄二值遮罩的欄位名稱，供 `generate_candidates`、`tracking`、`family_tree_query` 共享。
- `normalize_shape_tuple(shape)`：把不同型別的 shape 表示（ndarray/list/tuple/int）轉成 tuple，避免 I/O 轉換時 loss。
- `encode_mask(mask)` / `pack_binary_mask(mask, full_resolution_shape=None)`：將 bool mask 壓縮成 packed bits（支援 base64 或 ndarray 型態），用於 JSON manifest 與 NPZ 存放。
- `is_packed_mask(entry)` / `unpack_binary_mask(entry)`：在讀取時自動判斷 payload 是否為 packed 型態並還原為 boolean ndarray。
- `downscale_binary_mask(mask, ratio)`：利用 PIL box filter 將 mask 縮放後再閾值化，`generate_candidates` 及 `tracking` 在執行 mask downscale 時會呼叫。

## `geometry.py`
- `bbox_from_mask_xyxy(mask)`：回傳 `(x_min, y_min, x_max, y_max)`，供 `generate_candidates`/`tracking` 產生 bounding boxes。
- `bbox_xyxy_to_xywh(bounds)`：轉換座標為 `(x, y, w, h)`，在與 SAM/SAM2 API 互動時使用。

> 這些函數都是 `my3dis.common_utils` 所 re-export 的基礎原件，任何 mask 處理邏輯都會依賴這裡的實作。EOF
