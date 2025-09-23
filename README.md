3DIS Pipeline (Semantic-SAM × SAM2)

Summary
- Replace SAM in Algorithm 1 with Semantic-SAM and run the modified tracker per Semantic-SAM level.
- Drive multi-level Semantic-SAM at fixed levels [2,4,6] by default, sampling frames via ranges such as 1200:1600:20, and throttle expensive SSAM calls with `--ssam-freq` when desired.
- Persist raw candidate lists, filtered mask metadata (packed into JSON), SAM2 tracking masks, and streamlined comparison visuals while automatically gap-filling large uncovered regions at the coarsest level.
- Execute the pipeline through two dedicated conda environments (Semantic-SAM + SAM2); use `run_experiment.sh` to orchestrate stage switching and propagate the shared knobs (levels, frame slice, thresholds, SSAM cadence, SAM2 propagation limit).

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
- Conda envs: `Semantic-SAM` (Detectron2 0.6 + Torch 1.13) and `SAM2` (Torch 2.x). The single-script pipeline still requires a unified environment, but the recommended flow swaps between these two envs.

Data & Selection
- Source frames come from /media/public_dataset2/multiscan/<scene>/outputs/color (do not write to this location).
- Frame sampling uses Python slice syntax `start:end:step` (end exclusive). The canonical range is 1200:1600:20; the demo often uses the first three sampled frames for quick checks.
- Selected frames are symlinked or copied into `selected_frames/` inside each run directory for traceability.

Pipeline Overview
1) Frame selection: collect frames per the slice range and create a subset directory.
2) Semantic-SAM proposal generation (per level): run progressive refinement for each level, computing segmentation, bbox (XYWH), area, stability score, and metadata; any uncovered regions above `min_area` are converted into gap-fill masks at the coarsest level so tracking always receives an explicit prompt.
3) SAM2 tracking: prompt SAM2 with filtered masks/boxes, propagate (optionally capped by `--sam2-max-propagate`) to build masklets, and merge per-object masks per absolute frame index.
4) Persistence & visualization: store raw candidates, filtered summaries with packed masks inside JSON, propagated masks (`video_segments.npz`), per-object metadata, and comparison panels (overlays/instance maps are now skipped by default).

Outputs & Artifacts
- Default root: `My3DIS/outputs/<scene>/<timestamp>/` (use `--no-timestamp` to override).
- Level folder layout: `candidates/`, `filtered/`, `tracking/`, `viz/` plus shared `selected_frames/` at the run root.
- `candidates/candidates.json` keeps raw proposal metadata; `filtered/filtered.json` embeds filtered masks (packed bits + shape) directly in JSON, so no extra `.npy` files are saved.
- `tracking/video_segments.npz` stores propagated masks; `tracking/objects/L<L>_ID<id>/metadata.json` summarizes per-object stats alongside an `objects/index.json` roster (no PNG renders by default).
- `viz/compare/` holds contrast panels for Semantic-SAM vs. SAM2; other overlay/instance-map renders are disabled to minimise runtime and disk usage.
- Each run writes `manifest.json` with frame selection, thresholds, model paths, timestamps, the SSAM subset (`ssam_frames`, `ssam_freq`), and any SAM2 propagation cap in effect.

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
  conda run -n Semantic-SAM \
    python My3DIS/generate_candidates.py \
      --data-path /media/public_dataset2/multiscan/scene_00065_00/outputs/color \
      --levels 2,4,6 \
      --frames 1200:1600:20 \
      --ssam-freq 2 \
      --output /media/Pluto/richkung/My3DIS/outputs/scene_00065_00
  ```
- Track with SAM2 (SAM2 env)
  ```bash
  TS=$(ls -1dt /media/Pluto/richkung/My3DIS/outputs/scene_00065_00/* | head -n1)
  conda run -n SAM2 \
    python My3DIS/track_from_candidates.py \
      --data-path /media/public_dataset2/multiscan/scene_00065_00/outputs/color \
      --candidates-root "$TS" \
      --output "$TS" \
      --sam2-max-propagate 30
  ```
- (Optional) Build per-frame containment hierarchy
  ```bash
  python My3DIS/build_hierarchy.py \
    --candidates-root "$TS" \
    --levels 2,4,6 \
    --contain-thr 0.98
  ```

Single-Environment Option (advanced)
- If you maintain a single environment that imports both stacks, `run_pipeline.py` still provides an all-in-one path. Otherwise prefer the two-stage flow above.

Notes & Tips
- The tracker prefers mask prompts (`add_new_mask`) for fidelity; boxes are a fallback when masks are missing.
- SAM2 logits thresholding defaults to >0.0. Adjusting to 0.4–0.6 can sharpen edges—add a CLI flag if needed.
- `selected_frames/` captures the exact frames passed to SAM2, aiding debugging and reproducibility.
- Outputs are `.gitignore`d; commit code/configs, or add representative samples selectively.

Implementation Notes
- Stage 1 (`generate_candidates.py` + `ssam_progressive_adapter.py`): runs Semantic-SAM progressive refinement per level, throttled by `--ssam-freq`, synthesises base-level gap-fill masks for uncovered regions ≥ `min_area`, and keeps progressive outputs in temporary directories (no `_progressive_tmp` folder under the run root).
- Stage 2 (`track_from_candidates.py`): seeds SAM2 with filtered masks/boxes, respects `--sam2-max-propagate` to cap forward/backward steps, and renders only the SSAM-processed frames for consistency.
- `run_experiment.sh` wires both stages together, forwarding shared flags so a single CLI controls cadence, thresholds, and propagation depth across environments.

實作說明（繁體中文版）
- 第一階段（`generate_candidates.py` 與 `ssam_progressive_adapter.py`）：針對指定的 SSAM 取樣頻率執行 progressive refinement，並在最粗層自動補上大於 `min_area` 的未覆蓋區域，使後續追蹤一定有對應的遮罩。
- 第二階段（`track_from_candidates.py`）：以篩選後的遮罩／方框提示 SAM2，依 `--sam2-max-propagate` 限制向前向後的傳播步數，僅渲染真正做過 SSAM 的影格以保持輸出一致。
- `run_experiment.sh` 串接兩個環境，將層級、時間取樣、SSAM 頻率與 SAM2 傳播深度等參數一次傳遞，方便透過同一個指令調整流程。

Project Files
- `My3DIS/generate_candidates.py` — Stage 1 wrapper around Semantic-SAM progressive refinement.
- `My3DIS/track_from_candidates.py` — Stage 2 SAM2 tracker producing masks and visualizations.
- `My3DIS/run_pipeline.py` — Single-script pipeline bridging both stacks.
- `My3DIS/build_hierarchy.py` — Optional mask containment post-processing.
- `My3DIS/ssam_progressive_adapter.py` — Adapter utility for Semantic-SAM calls.
- `My3DIS/Algorithm1_env.yml` — Reference environment spec.
- `My3DIS/Agent.md` — Project log and status updates (operational details now live here in the README).
