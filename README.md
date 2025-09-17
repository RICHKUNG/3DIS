3DIS Pipeline (Semantic-SAM × SAM2)

Summary
- Replace SAM in Algorithm 1 with Semantic-SAM and run the modified tracker per Semantic-SAM level.
- Drive multi-level Semantic-SAM at fixed levels [2,4,6] by default, sampling frames via ranges such as 1200:1600:20.
- Persist raw candidate lists, filtered candidates, SAM2 tracking masks, and lightweight visualizations for reproducibility.
- Execute the pipeline through two dedicated conda environments (Semantic-SAM + SAM2); use `run_experiment.sh` to orchestrate stage switching automatically.

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
2) Semantic-SAM proposal generation (per level): run progressive refinement for each level, computing segmentation, bbox (XYWH), area, stability score, and metadata.
3) SAM2 tracking: prompt SAM2 with filtered masks/boxes, propagate to build masklets, and merge per-object masks per absolute frame index.
4) Persistence & visualization: store raw candidates, filtered summaries with `.npy` mask stacks, propagated masks (`video_segments.npz`), per-object masked imagery, overlays, instance maps, and comparison panels.

Outputs & Artifacts
- Default root: `My3DIS/outputs/<scene>/<timestamp>/` (use `--no-timestamp` to override).
- Level folder layout: `candidates/`, `filtered/`, `tracking/`, `viz/` plus shared `selected_frames/` at the run root.
- `candidates/candidates.json` keeps raw proposal metadata; `filtered/filtered.json` plus `seg_frame_*.npy` hold filtered masks.
- `tracking/video_segments.npz` stores propagated masks; `tracking/objects/L<L>_ID<id>/*.png` keeps labeled masked images.
- `viz/` contains color overlays, instance maps (`instance_map/*.png` & `.npy`), and `compare/` panels contrasting Semantic-SAM vs. SAM2.
- Each run writes `manifest.json` with frame selection, thresholds, model paths, and timestamps for reproducibility.

Execution

Recommended: Orchestrated Two-Stage Run
- `run_experiment.sh` switches between the two conda envs and calls both Python stages with the baked-in checkpoints and fixed scene/output paths (edit the constants at the top of the script to target another scene).
  ```bash
  cd /media/Pluto/richkung/My3DIS
  ./run_experiment.sh --levels 2,4,6 --frames 1200:1600:20
  ```
- Add `--dry-run` to inspect commands without executing, `--no-timestamp` to write directly into the fixed output directory, and threshold overrides via `--min-area` / `--stability`. Checkpoint/config overrides remain available if needed.

Manual Two-Stage Flow
- Generate candidates (Semantic-SAM env)
  ```bash
  conda run -n Semantic-SAM \
    python My3DIS/generate_candidates.py \
      --data-path /media/public_dataset2/multiscan/scene_00065_00/outputs/color \
      --levels 2,4,6 \
      --frames 1200:1600:20 \
      --output /media/Pluto/richkung/My3DIS/outputs/scene_00065_00
  ```
- Track with SAM2 (SAM2 env)
  ```bash
  TS=$(ls -1dt /media/Pluto/richkung/My3DIS/outputs/scene_00065_00/* | head -n1)
  conda run -n SAM2 \
    python My3DIS/track_from_candidates.py \
      --data-path /media/public_dataset2/multiscan/scene_00065_00/outputs/color \
      --candidates-root "$TS" \
      --output "$TS"
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

Project Files
- `My3DIS/generate_candidates.py` — Stage 1 wrapper around Semantic-SAM progressive refinement.
- `My3DIS/track_from_candidates.py` — Stage 2 SAM2 tracker producing masks and visualizations.
- `My3DIS/run_pipeline.py` — Single-script pipeline bridging both stacks.
- `My3DIS/build_hierarchy.py` — Optional mask containment post-processing.
- `My3DIS/ssam_progressive_adapter.py` — Adapter utility for Semantic-SAM calls.
- `My3DIS/Algorithm1_env.yml` — Reference environment spec.
- `My3DIS/Agent.md` — Project log and status updates (operational details now live here in the README).
