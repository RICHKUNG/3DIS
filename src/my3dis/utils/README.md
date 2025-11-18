# `utils/` 模組

| 檔案 | 函數 | 摘要 |
| --- | --- | --- |
| `logging.py` | `setup_logging(explicit_level=None, env_var='MY3DIS_LOGLEVEL', logger_names_to_quiet=())` | 初始化 Python logging，設定 format、等級並靜音第三方 loggers；`configure_entry_log_format()` 進一步把 PID/時間戳加入 CLI entrypoint。`run_workflow.py`、`generate_candidates.py` 等入口皆呼叫這裡。 |
| `parsing.py` | `list_to_csv(values)`、`parse_levels(level_spec)`、`parse_range(range_str)` | 將 YAML/CLI 參數轉成統一格式：清單轉 CSV 字串、層級字串轉整數清單、`start:end:step` 轉 slice。`generate_candidates`、`track_from_candidates` 與工作流程設定皆依賴這些 helper。 |
| `ply_utils.py` | `_resolve_scene_ply(path)`、`load_scene_pointcloud(scene_root)`、`save_mask_as_ply(points, mask, output_path)` | 提供多邊形/點雲載入與匯出。Aggregation 及 SigLIP 模組在需要把遮罩寫成 `.ply` 檔或載入場景點雲時使用。 |
| `sorting.py` | `numeric_frame_sort_key(filename)` | 以內嵌數字排序影像檔案名稱，避免 `frame_10` 排在 `frame_2` 前面，由 `generate_candidates`、`track_from_candidates`、`family_tree_builder` 共享。 |
| `time_utils.py` | `format_duration(seconds)` | 將秒數格式化成 `H:MM:SS` 或 `M:SS`，應用在 workflow summary、report、tracking log 等。 |

> 這些函數也會由 `my3dis/common_utils.py` re-export，供其他模組統一匯入入口。EOF
