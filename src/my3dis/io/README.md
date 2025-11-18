# `io/` 模組

## `fs_utils.py`
- `ensure_dir(path)`：建立目標資料夾（含父層）並回傳絕對路徑字串，供 `generate_candidates`、`track_from_candidates` 等模組在寫入前確保目錄存在。
- `build_subset_video(frames_dir, selected, selected_indices, out_root, folder_name='selected_frames')`：依照挑選出的幀號建立精簡版影像資料夾，對原始影像建立 symlink（若檔案系統不支援則複製）。回傳 `(subset_dir, {abs_idx: subset_filename})`，提供 SAM2 追蹤器與比較視覺化使用。

> 其餘子模組位於 `my3dis/utils` 與 `my3dis/common_utils`，這裡僅存放純檔案系統操作。EOF
