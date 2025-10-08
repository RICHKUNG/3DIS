3DIS Pipeline Log (Semantic-SAM × SAM2)

Reference
- All pipeline goals, environment details, dataset policy, and execution guidance now live in `README.md`.
- This file tracks decisions, progress, and outstanding work items.

Status
- Plan agreed: levels [2,4,6], frame slice 1200:1600:20, quick demos use the first three sampled frames.
- `scripts/pipeline/run_pipeline.py` now reuses the same candidate/track modules as `run_experiment.sh`, so a single-environment run yields identical artifacts without duplicating logic; the default workflow still swaps between dedicated Semantic-SAM and SAM2 conda envs.
- `run_experiment.sh` coordinates the two-stage execution (generate → track); dataset/output paths are fixed in the script so the CLI only adjusts levels, frame ranges, thresholds, Semantic-SAM cadence, and SAM2 propagation depth.
- Progressive refinement now back-fills large uncovered regions at the coarsest level, so SAM2 always receives explicit masks for gaps that exceed the `fill_area` threshold (defaulting to `min_area`).
- Per-level outputs must persist raw and filtered candidate lists alongside SAM2 tracking artifacts.
- YAML 驅動的 `src/my3dis/run_workflow.py` 可依配置自動執行 SSAM → filter → SAM2 → 報告，並在 YAML 中指定 GPU/參數、產出 Markdown 紀錄。
- `scripts/pipeline/run_workflow_batch.py` + `configs/scenes/` 已覆蓋 MultiScan 全場景，並自動將執行摘要寫入 `logs/workflow_history.csv` 與批次報表，方便規劃跨場景 sweep。
- YAML `experiment.scenes` + `experiment.dataset_root` 現在支援單一實驗跑多個場景，輸出集中在 `outputs/experiments/<experiment>/<scene>/...`。
- `configs/multiscan/base.yaml` 集中定義跨場景 sweep（預設 `scenes: all`），`src/my3dis/run_workflow.py` 會自動建立實驗 timestamp 資料夾並整理輸出成 scene/level 目錄、`summary.json`、三張報表圖片與 SAM2 NPZ。
- `src/my3dis/run_workflow.py` 支援 `scene_start`/`scene_end`，可用來鎖定 MultiScan 範圍，例如 `scene_start: scene_00030_00`、`scene_end: scene_00075_01` 會按照資料夾順序依序執行。

Progress Log
- Added `scripts/pipeline/run_pipeline.py` to glue Semantic-SAM candidate generation with SAM2 tracking.
- Added `src/my3dis/generate_candidates.py` (Semantic-SAM stage) and `src/my3dis/track_from_candidates.py` (SAM2 stage) for two-environment execution.
- Added `run_experiment.sh` to orchestrate stage execution across environments with sensible defaults.
- Completed demo run on first three frames of scene_00065_00 via the two-stage path; results saved under `My3DIS/outputs/scene_00065_00/demo`.
- README updated with consolidated operational guidance from the earlier planning document, plus the fixed-path orchestrator workflow.
- 2025-09-18: Unified pipeline code paths (run_pipeline → generate_candidates/track_from_candidates, progressive candidates now honor stability thresholds) and executed `./run_experiment.sh --levels "1,3,5" --frames "1200:1600:20" --min-area 400 --stability 0.8`; full run finished in ~22m with outputs under `My3DIS/outputs/scene_00065_00/20250918_095823`.
- 2025-09-18: Investigated why `python3 scripts/pipeline/run_pipeline.py` is slower than `./run_experiment.sh`; root cause is single-environment execution missing Semantic-SAM/SAM2 specific CUDA builds and the combined process double-buffering both models in memory. Recommendation kept: prefer `run_experiment.sh` or reintroduce per-stage environments when running the Python orchestrator directly.
- 2025-09-22: Added `--ssam-freq` to throttle Semantic-SAM usage and record the subset of frames that actually run segmentation; `src/my3dis/generate_candidates.py` now only builds artifacts for those frames.
- 2025-09-22: Added SAM2 propagation cap (`--sam2-max-propagate`) throughout CLI/orchestrators and reworked tracking calls to honor frame budgets without hitting the legacy API signature.
- 2025-09-22: Implemented gap-fill mask synthesis in `src/my3dis/ssam_progressive_adapter.py` so uncovered regions larger than the `fill_area` threshold (then aligned with `min_area`) enter downstream filtering/tracking even if Semantic-SAM misses them in the first pass.
- 2025-09-23: Streamlined persistence—progressive refinement now runs inside temp dirs (no `_progressive_tmp` artifacts), filtered masks are packed into `filtered.json`, tracking objects emit JSON-only metadata, and viz renders only keep the `compare/` panels.
- 2025-09-24: Reviewed recent SAM2 tracking failures; noted ~5k mask prompts per long run (logs `nohupGPU0.out`/`nohupGPU1.out`) which translate to ~24–36 GB of GPU tensor memory because each mask prompt persists as a 1024² float tensor. Advised throttling mask additions (filtering, chunked propagation, or converting to boxes) to stay within GPU limits.
- 2025-09-25: Added CLI overrides for SAM2 IoU threshold and box-prompt policies (`run_experiment.sh` → `src/my3dis/track_from_candidates.py`); long-tail objects default to area ≤ max(3×min_area, min_area+1) with `MY3DIS_LONG_TAIL_AREA` override, and scripts log the active prompt strategy.
- 2025-09-26: 建立 `configs/scene_00065_00.yaml` 與 `src/my3dis/run_workflow.py`，將 SSAM/Filter/SAM2/報告拆成 stage，可在 YAML 中調整 level、frame freq、SSAM freq、GPU 配置；新增 `src/my3dis/filter_candidates.py`、`src/my3dis/generate_report.py` 支援重複篩選與 Markdown 報告（含每層第一/中位/最後 compare 圖、時間摘要）。
- 2025-09-26: Refined tracking artifacts—gap-fill僅在第一個 level 啟用、SAM2 僅輸出 frame/object `.npz` 成對檔案、viz 比較圖改為每 10 張 SSAM 幀儲存；README / Agent 記錄同步更新。
- 2025-09-27: Tracker 支援遮罩縮放開關（YAML `downscale_masks` + `downscale_ratio`），SAM2/SSAM 遮罩可縮至 0.3× 後再封裝，輸出的 `.npz` 以 `_scale{ratio}x` 後綴標示並記錄原始尺寸以供還原。
- 2025-09-27: 匯出 MultiScan 場景 config (`configs/scenes/*.yaml`) 與 `configs/index/multiscan_scene_index.json`，新增 `scripts/pipeline/run_workflow_batch.py` 與 `scripts/prepare_scene_configs.py`，workflow 結束時自動寫入 `logs/workflow_history.csv`／`logs/batch/*.json`。
- 2025-09-27: `src/my3dis/run_workflow.py` 支援 `experiment.scenes` 多場景執行，將 `experiment.output_root` 當作實驗根目錄並在 `/<scene>/` 建立 run；History CSV 新增 `parent_experiment` / `scene_index` 欄位，`scripts/pipeline/run_workflow_batch.py` 會展開每個場景的紀錄。
- 2025-10-02: 新增 `configs/multiscan/base.yaml`，`src/my3dis/run_workflow.py` 自動列舉 MultiScan 全場景、建立 `aggregate_output` timestamp 根目錄，SAM2 階段支援 `render_viz` 關閉，報告輸出整併為每個 `scene/level` 目錄含 `object_segments_Lxx.npz`、`video_segments_Lxx.npz`、三張報表圖片與 `summary.json`。
- 2025-10-08: Added `--dry-run` flag to `src/my3dis/run_workflow.py`, documented the `PYTHONPATH` export requirement in README, and verified with `PYTHONPATH=src conda run -n My3DIS python -m my3dis.run_workflow --config configs/scenes/scene_00065_00.yaml --dry-run`, which now prints the parsed config without executing stages.
- 2025-10-07: 清理 `configs/multiscan_all.yaml` 合併衝突並改為 `persist_raw=true` / `skip_filtering=true`，同時重新啟用 filter stage（沿用 `min_area=1000`, `stability_threshold=1.0`）以將 SSAM 遮罩改為磁碟暫存，避免 OOM。
- 2025-10-07: 導入 `oom_monitor/` 工具集，`src/my3dis/run_workflow.py` 與 `scripts/pipeline/run_workflow_batch.py` 預設掛載 cgroup `memory.events` watcher，將 OOM 指標寫入 `logs/oom_monitor.log`（`--no-oom-watch` 可停用，`--oom-watch-*` 旗標可調整行為）。

Next Actions
1) Create the shared environment from `archive/env/Algorithm1_env.yml` (optional but recommended).
2) Update or clone the notebook (`archive/experiments/algorithm1.ipynb`) to reuse the helper functions if interactive analysis is needed.
3) Execute the refreshed pipeline on a larger slice using the new `--ssam-freq` / `--sam2-max-propagate` knobs to validate runtime savings. ✅ small-scale verification pending.
4) Push code to https://github.com/RICHKUNG/3DIS when credentials are ready.

Open Items
- 啟動第一批跨場景 batch（透過 `scripts/pipeline/run_workflow_batch.py`），並確認 `logs/workflow_history.csv`、`logs/batch/*.json` 的內容完整。
- Tuning knobs: `min_area`, `fill_area`, `stability_threshold`, and SAM2 IoU threshold (currently 0.6).
- Validate the 0.3× mask persistence path on a longer scene（確保 `_scale{ratio}x` 輸出仍可還原與生成報表）。

GitHub
- Initialize/push sequence from `My3DIS/`:
  - `git init && git checkout -b main`
  - `git remote add origin https://github.com/RICHKUNG/3DIS.git`
  - `git add Agent.md scripts/pipeline/run_pipeline.py archive/env/Algorithm1_env.yml archive/experiments/algorithm1.ipynb`
  - `git commit -m "Init: multi-level Semantic-SAM → SAM2 pipeline"`
  - `git push -u origin main`
