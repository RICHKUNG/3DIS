3DIS Pipeline (Semantic-SAM × SAM2)

Summary
- Replace SAM in Algorithm 1 with Semantic-SAM and run the modified tracker per Semantic-SAM level.
- Drive multi-level Semantic-SAM at fixed levels [2,4,6] by default, sampling frames via ranges such as 1200:1600:20 (CLI) or by setting only `step` in YAML to span the full sequence, and throttle expensive SSAM calls with `--ssam-freq` when desired.
- Persist raw candidate lists, filtered mask metadata (packed into JSON), SAM2 tracking `.npz` artifacts (frame-major + object-major, suffixed with `_scale{ratio}x` when mask downsampling is enabled), and comparison visuals while automatically gap-filling large uncovered regions on the coarsest level only when `add_gaps=true`.
- Execute the pipeline through two dedicated conda environments (Semantic-SAM + SAM2); use `run_experiment.sh` or invoke the YAML-driven orchestrator via `python -m my3dis.run_workflow --config <path>` (set `PYTHONPATH=src` when running from the repo checkout) to propagate shared knobs (levels, frame slice, thresholds, SSAM cadence, SAM2 propagation limit, prompt 策略等)。
- Workflow history appends now acquire file locks, so parallel runs safely extend `logs/workflow_history.csv` without corrupting headers or rows。
- 透過 `src/my3dis/generate_report.py` 自動輸出 Markdown 報告，記錄階段耗時、參數設定與各層代表性的 Semantic-SAM / SAM2 比對圖，有助於後續調參與結果彙整。
- 匯出 MultiScan 全場景的 YAML 設定至 `configs/scenes/`，可透過 `scripts/prepare_scene_configs.py` 更新，並利用 `scripts/pipeline/run_workflow_batch.py` 進行跨場景批次執行；每次 workflow 會同步記錄於 `logs/workflow_history.csv`。
- YAML `experiment.scenes` 搭配 `experiment.dataset_root` 可一次指定多個場景，輸出會集中放在 `experiment.output_root/<scene>/...`，最外層依實驗命名。

Goals
- Produce per-level mask candidates with Semantic-SAM for a chosen frame range.
- Track unassigned regions using SAM2 mask propagation (masklets) and merge them with the Semantic-SAM proposals.
- Deliver artifacts that allow the entire pipeline to be re-run or inspected offline.

Environment
- Dataset (read-only): /media/public_dataset2/multiscan/<scene>/outputs/color
- Repos: Semantic-SAM at /media/Pluto/richkung/Semantic-SAM; SAM2 at /media/Pluto/richkung/SAM2
- Checkpoints: defaults are baked into the Python scripts and shell helper:
  - Semantic-SAM SwinL → /media/Pluto/richkung/Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth
  - SAM2 config → /media/Pluto/richkung/SAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
  - SAM2 weights → /media/Pluto/richkung/SAM2/checkpoints/sam2.1_hiera_large.pt
- Environment overrides: export `MY3DIS_SEMANTIC_SAM_ROOT`, `MY3DIS_SEMANTIC_SAM_CKPT`, `MY3DIS_SAM2_ROOT`,
  `MY3DIS_SAM2_CFG`, `MY3DIS_SAM2_CKPT`, `MY3DIS_DATASET_ROOT`, `MY3DIS_DATA_PATH`, or `MY3DIS_OUTPUT_ROOT`
  to redirect repositories, checkpoints, dataset roots, and output folders without editing YAML; CLI overrides
  keep the highest precedence where available.
- Conda envs: `Semantic-SAM` (Detectron2 0.6 + Torch 1.13) and `SAM2` (Torch 2.x). The single-script pipeline still requires a unified environment, but the recommended flow swaps between these two envs.
- Environment manifests live in `env/` (current exports) while frozen snapshots remain in `archive/env/`.
- Legacy notebooks and auxiliary scripts now sit under `archive/experiments/` for reference.
- When running modules directly (`python -m my3dis.*`), export `PYTHONPATH=$(pwd)/src` or prefix commands with `PYTHONPATH=src` so the package is discoverable.

Data & Selection
- Source frames come from /media/public_dataset2/multiscan/<scene>/outputs/color (do not write to this location).
- Frame sampling accepts Python slice syntax `start:end:step` (end exclusive). When using `src/my3dis/run_workflow.py`, omitting `start`/`end` and keeping only `step` automatically spans every available frame; the canonical demo slice remains 1200:1600:20 for quick CLI tests.
- Selected frames are symlinked or copied into `selected_frames/` inside each run directory for traceability.
- Example metadata (e.g., `scene_00065_00_superpoint_pixel_locations.json`) now lives in `data/examples/` for quick reference without polluting outputs.
- `configs/index/multiscan_scene_index.json` 彙整所有場景、影格總數與對應的 YAML 路徑，方便挑選與查詢。
- Multi-scene YAML 指定 `experiment.dataset_root`（如 `/media/public_dataset2/multiscan`）與 `experiment.scenes` 名稱列表，系統會自動解析 `outputs/color` 子資料夾並於 `experiment.output_root/<scene>/` 建立執行結果；若需要快速切換資料磁碟，可先匯出 `MY3DIS_DATASET_ROOT` / `MY3DIS_OUTPUT_ROOT` 以覆寫設定值。

Pipeline Overview
1) Frame selection: collect frames per the slice range and create a subset directory.
2) Semantic-SAM proposal generation (per level): run progressive refinement for each level, computing segmentation, bbox (XYWH), area, stability score, and metadata; when `add_gaps=true`, uncovered regions above the `fill_area` threshold (defaulting to `min_area`) are converted into gap-fill masks for the first (typically coarsest) level only, guaranteeing at least one prompt for large holes without duplicating them across scales.
3) SAM2 tracking: prompt SAM2 with filtered masks/boxes, propagate (optionally capped by `--sam2-max-propagate`) to build masklets, and merge per-object masks per absolute frame index.
4) Filtering (optional stage): 如在 SSAM 階段啟用 raw 儲存 (`persist_raw=true`)，可在後續任意時間透過 `src/my3dis/filter_candidates.py` 重新套用不同的 `min_area`、`stability` 門檻，而不必重跑 Semantic-SAM。
5) Persistence, report & visualization: store raw candidates, filtered summaries with packed masks inside JSON, propagated masks (`video_segments*.npz` + `object_segments*.npz`, where a `_scale{ratio}x` suffix denotes binary masks stored at reduced resolution), sparsified comparison panels（預設僅保留每第 10 個 SSAM 幀），以及由 `src/my3dis/generate_report.py` 產生的 Markdown 報告（含代表性縮圖與各階段耗時）。

Workflow Orchestrators

**YAML 驅動 (`src/my3dis/run_workflow.py`)**
- 透過 `configs/*.yaml` 定義實驗：`experiment` 區塊指定資料來源、輸出根目錄、預設 levels，並可用 `run_dir` 直接指向既有的 SSAM 輸出以跳過重跑；`stages` 區塊逐段調整是否啟用、GPU ID、取樣頻率、prompt 模式（`none` / `long_tail` / `all`）、最大傳播步數等。
- 新增的 `StageRecorder` 會在 `workflow_summary.json` 記錄每個 stage 的 GPU、開始/結束時間與耗時，供 `src/my3dis/generate_report.py` 匯整。
- 若 `experiment.scenes` 列出多個場景，`src/my3dis/run_workflow.py` 會依序執行並在 `experiment.output_root/<scene>/` 寫入 timestamp 子資料夾，同時在 summary 與 `workflow_history.csv` 中附上 `parent_experiment` 與 `scene_index`。
- 示例：
  ```bash
  # 在具備 Torch + Semantic-SAM + SAM2 的環境中執行
  cd /media/Pluto/richkung/My3DIS
  export PYTHONPATH="$(pwd)/src"
  nohup python3 -m my3dis.run_workflow \
    --config configs/scene_00065_00.yaml \
    > logs/workflow_$(date +%Y%m%d_%H%M).log 2>&1 &
  ```
  *(append `--dry-run` to print the parsed config without executing)*
  - `stages.tracker.prompt_mode` 選項：`all_mask`（僅用 mask prompt）、`lt_bbox`（小面積用 bbox）、`all_bbox`（全部用 bbox）。
  - `stages.tracker.downscale_masks=true` 會在追蹤階段將 SAM2/SSAM mask 以 `downscale_ratio`（預設 0.3）等比例縮小後再持久化，輸出的 `.npz` 會附上 `_scale{ratio}x` 後綴以標示倍率。
  - `filter` stage 會自動檢查 `level_*/raw` 是否存在，若尚未啟用 `persist_raw` 則跳過並在 summary 標記 `skipped=missing_raw`。
  - 報告 (`report.md`) 與縮圖輸出在各 `level_x/report/` 底下，方便比較不同 run。

**批次執行 (`scripts/pipeline/run_workflow_batch.py`)**
- 從指定目錄（預設 `configs/scenes/`）蒐集 YAML，依序呼叫 `my3dis.run_workflow.execute_workflow`，並在 `logs/batch/batch_<timestamp>.json` 寫入整體批次摘要。
- 每個 workflow 完成後會於 `logs/workflow_history.csv` 追加一列記錄（scene、levels、frame 統計、執行參數等），方便後續查詢與彙整。
- 常見指令：
  ```bash
  # 依序跑完整個 MultiScan 清單（使用預設 config 目錄）
  python3 scripts/pipeline/run_workflow_batch.py --config-dir configs/scenes

  # 僅針對指定場景
  python3 scripts/pipeline/run_workflow_batch.py --config-dir configs/scenes --scenes scene_00065_00 scene_00075_00

  # 先查看預計執行的前三個場景
  python3 scripts/pipeline/run_workflow_batch.py --config-dir configs/scenes --limit 3 --dry-run
  ```
- 若需重新匯出全套 config，可執行 `python3 scripts/prepare_scene_configs.py --project-root .`（支援 `--scenes`、`--skip-existing` 等參數）。

**雙環境 Shell (`run_experiment.sh`)**
- 維持原先的兩段指令流程，適合沒有統一環境的情境；同樣支援 `--ssam-freq`、`--sam2-max-propagate`、`--min-area`、`--fill-area` 等參數。

- Default root: `My3DIS/outputs/<scene>/<timestamp>/` (use `--no-timestamp` to override)。`src/my3dis/run_workflow.py` 會在根目錄另外寫入 `workflow_summary.json` 與 `report.md`。
- Level folder layout: `candidates/`, `raw/`（若有啟用持久化）、`filtered/`, `tracking/`, `viz/`, `report/`。
- `candidates/candidates.json` keeps raw proposal metadata；若啟用 `persist_raw=True`，`raw/frame_XXXXX.json` + `raw/frame_XXXXX.npz` 會保存完整遮罩堆疊，供 `src/my3dis/filter_candidates.py` 重跑。
- `filtered/filtered.json` embeds filtered masks (packed bits + shape) directly in JSON，若重新套用篩選則會覆寫此檔案並更新 manifest。
- `tracking/video_segments*.npz` 以 manifest 方式封裝（`manifest.json` + `frames/frame_XXXXX.json`），每個 frame 檔案包含經過 base64 編碼的 packed mask，並涵蓋 `experiment.frames.step` 抽出的每張影格（即使 `ssam_freq > 1`，SAM2 仍會輸出完整序列）；`tracking/object_segments*.npz` 轉為 object → frame 的參考表，避免重複寫入遮罩資料，讀取時可透過 `frame_entry` 反查 `video_segments` 中的對應 JSON。
- `viz/compare/` holds contrast panels for Semantic-SAM vs. SAM2；輸出已稀疏化為每第 10 個 SSAM 幀。`level_x/report/` 底下仍保留縮小後的代表性圖檔（第一張／中位／最後一張），供 Markdown 報告引用。
- `report.md`（根目錄）為中文摘要，包含階段耗時表格、主要參數、每個 level 的代表圖表連結；`workflow_summary.json` 保存原始紀錄。
- Each run writes `manifest.json` with frame selection, thresholds, model paths, timestamps, the SSAM subset (`ssam_frames`, `ssam_freq`), any SAM2 propagation cap in effect，以及篩選/原始資料的設定。
- `prepare_tracking_run.py` 可以複製既有的 SSAM run（支援 hardlink 或深拷貝）並更新 YAML 的 `experiment.run_dir`，方便用同一份 candidates 重複測試不同的 tracker 參數。

完整輸出目錄示例（略去大型檔案）：
```
outputs/experiments/multiscan_demo/scene_00065_00/2025xxxx_xxxxxx/
├── manifest.json
├── workflow_summary.json
├── report.md
├── selected_frames/
├── level_2/
│   ├── candidates/candidates.json
│   ├── raw/frame_01200.json + .npz   # 若 persist_raw
│   ├── filtered/filtered.json
│   ├── tracking/video_segments_scale0.3x.npz
│   ├── tracking/object_segments_scale0.3x.npz
│   ├── viz/compare/frame_01200_L2.png
│   └── report/frame_01200_第一張.png
├── level_4/
│   └── ...
└── level_6/
    └── ...
```

Filter & Report Utilities

- `src/my3dis/filter_candidates.py`：重複篩選存放於 `raw/` 的 Semantic-SAM 遮罩，支援 `--levels`、`--min-area`、`--stability-threshold` 與 `--update-manifest`。
  ```bash
  python3 filter_candidates.py \
    --candidates-root outputs/experiments/multiscan_demo/scene_00065_00/<run> \
    --min-area 400 --stability-threshold 0.85 --update-manifest
  ```
- `src/my3dis/generate_report.py`：整理 `workflow_summary.json` 與 `manifest.json`，生成 Markdown 報告與代表性圖檔，可調整 `--max-width`。
  ```bash
  python3 generate_report.py --run-dir outputs/experiments/multiscan_demo/scene_00065_00/<run> --max-width 640
  ```

Execution

Recommended: Orchestrated Two-Stage Run
- `run_experiment.sh` switches between the two conda envs and calls both Python stages with the baked-in checkpoints and fixed scene/output paths (edit the constants at the top of the script to target another scene).
  ```bash
  cd /media/Pluto/richkung/My3DIS
  ./run_experiment.sh --levels 2,4,6 --frames 1200:1600:20 --ssam-freq 2 --sam2-max-propagate 30
  ```
- Add `--dry-run` to inspect commands without executing, `--no-timestamp` to write directly into the fixed output directory, tweak cadence via `--ssam-freq`, and bound propagation with `--sam2-max-propagate` alongside `--min-area` / `--stability`. Checkpoint/config overrides remain available if needed.

Manual Two-Stage Flow
- Generate candidates (Semantic-SAM env)
  ```bash
  export PYTHONPATH="$(pwd)/src"
  conda run -n Semantic-SAM \
    python -m my3dis.generate_candidates \
      --data-path /media/public_dataset2/multiscan/scene_00065_00/outputs/color \
      --levels 2,4,6 \
      --frames 1200:1600:20 \
      --ssam-freq 2 \
      --output /media/Pluto/richkung/My3DIS/outputs/scene_00065_00
  ```
- Track with SAM2 (SAM2 env)
  ```bash
  TS=$(ls -1dt /media/Pluto/richkung/My3DIS/outputs/scene_00065_00/* | head -n1)
  export PYTHONPATH="$(pwd)/src"
  conda run -n SAM2 \
    python -m my3dis.track_from_candidates \
      --data-path /media/public_dataset2/multiscan/scene_00065_00/outputs/color \
      --candidates-root "$TS" \
      --output "$TS" \
      --sam2-max-propagate 30
  ```
- (Optional) Build per-frame containment hierarchy
  ```bash
  python My3DIS/archive/experiments/build_hierarchy.py \
    --candidates-root "$TS" \
    --levels 2,4,6 \
    --contain-thr 0.98
  ```

Single-Environment Option (advanced)
- If you maintain a single environment that imports both stacks, `scripts/pipeline/run_pipeline.py` still provides an all-in-one path. Otherwise prefer the two-stage flow above.

Notes & Tips
- The tracker prefers mask prompts (`add_new_mask`) for fidelity; boxes are a fallback when masks are missing.
- SAM2 logits thresholding defaults to >0.0. Adjusting to 0.4–0.6 can sharpen edges—add a CLI flag if needed.
- `selected_frames/` captures the exact frames passed to SAM2, aiding debugging and reproducibility.
- `logs/workflow_history.csv` 會自動堆疊每次 `run_workflow` 執行摘要，`logs/batch/` 則保存批次報表，可納入後續分析或建置儀表板。
- Outputs are `.gitignore`d; commit code/configs, or add representative samples selectively.
- **Code Status**: Several optimization opportunities have been identified (large files, config unification, package installation) but current implementation remains stable and functional. See `PROBLEM.md` for improvement roadmap.

Implementation Notes
- Stage 1 (`src/my3dis/generate_candidates.py` + `src/my3dis/ssam_progressive_adapter.py`): runs Semantic-SAM progressive refinement per level, throttled by `--ssam-freq`, synthesises gap-fill masks ≥ `fill_area` (default = `min_area`) only on the first level when `add_gaps=true`, and keeps progressive outputs in temporary directories (no `_progressive_tmp` folder under the run root).
- Stage 2 (`src/my3dis/track_from_candidates.py`): seeds SAM2 with filtered masks/boxes, respects `--sam2-max-propagate` to cap forward/backward steps, optionally downsamples stored masks via `downscale_masks`/`downscale_ratio` (file名改為 `video_segments_scale{ratio}x.npz` / `object_segments_scale{ratio}x.npz`)，並僅輸出 frame/object `.npz` 與每 10 張 SSAM 幀的比較圖。
- `run_experiment.sh` wires both stages together, forwarding shared flags so a single CLI controls cadence, thresholds, and propagation depth across environments.

實作說明（繁體中文版）
- 第一階段（`src/my3dis/generate_candidates.py` 與 `src/my3dis/ssam_progressive_adapter.py`）：針對指定的 SSAM 取樣頻率執行 progressive refinement，並僅在第 1 個 level (`add_gaps=true` 時) 補上大於 `fill_area`（預設等同 `min_area`）的未覆蓋區域，避免同一缺口在多層重複出現。
- 第二階段（`src/my3dis/track_from_candidates.py`）：以篩選後的遮罩／方框提示 SAM2，依 `--sam2-max-propagate` 限制向前向後的傳播步數，可透過 `downscale_masks` / `downscale_ratio` 將遮罩縮小後儲存（輸出 `video_segments_scale{ratio}x.npz` / `object_segments_scale{ratio}x.npz`），並維持每第 10 個 SSAM 影格產生比較圖。
- `run_experiment.sh` 串接兩個環境，將層級、時間取樣、SSAM 頻率與 SAM2 傳播深度等參數一次傳遞，方便透過同一個指令調整流程。

- `My3DIS/src/my3dis/run_workflow.py` — YAML orchestrator，整合 SSAM → filter → SAM2 → 報告。
- `My3DIS/src/my3dis/filter_candidates.py` — 針對已儲存的 raw 遮罩重新套用篩選條件。
- `My3DIS/src/my3dis/generate_report.py` — 生成 Markdown 報告與縮圖。
- `My3DIS/src/my3dis/generate_candidates.py` — Stage 1 wrapper around Semantic-SAM progressive refinement。
- `My3DIS/src/my3dis/track_from_candidates.py` — Stage 2 SAM2 tracker producing masks and visualizations。
- `My3DIS/scripts/pipeline/run_pipeline.py` — 單一環境的備用流程。
- `My3DIS/archive/experiments/build_hierarchy.py` — Optional mask containment post-processing。
- `My3DIS/src/my3dis/src/my3dis/ssam_progressive_adapter.py` — Adapter utility for Semantic-SAM calls。
- `My3DIS/archive/env/Algorithm1_env.yml` — Reference environment spec。
- `My3DIS/Agent.md` — Project log and status updates (operational details now live here in the README)。
