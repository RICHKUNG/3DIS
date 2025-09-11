3DIS Pipeline Plan and Log (Semantic-SAM × SAM2)

Summary
- Replace SAM in Algorithm 1 with Semantic-SAM and run the modified Algorithm 1 per Semantic-SAM level.
- Use multi-level Semantic-SAM with fixed levels [2, 4, 6].
- Frame range: 1200:1600:20 (start:end:step, end exclusive).
- Demo first on the first 3 sampled frames; later run the full selection.
- Dataset: MultiScan, path is correct and read-only under /media/public_dataset2/multiscan. Do not modify anything under /media/public_dataset2.
- All code execution and outputs live under My3DIS.
- Save, for each level, both the raw candidate bbox lists and the filtered lists for reproducibility.

Goals
- Produce multi-level mask candidates using Semantic-SAM per frame and per level.
- For each level, run a modified Algorithm 1 that uses SAM2 to track untracked regions via mask-propagation (masklets) across the frame sequence.
- Provide visualizations and structured artifacts that allow exact reproduction.

Environment
- New conda environment created from My3DIS/Algorithm1_env.yml (CUDA 12.x toolchain per YAML; Torch 2.8.0+cu128 as specified).
- Local checkpoints:
  - Semantic-SAM: /media/Pluto/richkung/Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth
  - SAM2 config: /media/Pluto/richkung/SAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
  - SAM2 weights: /media/Pluto/richkung/SAM2/checkpoints/sam2.1_hiera_large.pt (or other variant)
- Import paths: either install editable or add to sys.path:
  - /media/Pluto/richkung/Semantic-SAM
  - /media/Pluto/richkung/SAM2

Data & Selection
- Source: /media/public_dataset2/multiscan/<scene>/outputs/color
- Frame sampling: 1200:1600:20 (end exclusive). Demo uses the first 3 sampled frames from this range.

Pipeline Overview
1) Frame selection: gather frames by the configured range.
2) Per-level Semantic-SAM generation (levels [2,4,6]):
   - Generate mask candidates on each selected frame using Semantic-SAM at the current level.
   - Each candidate includes at least segmentation (boolean array), area, stability_score (if available), and bbox (XYWH) computed from segmentation when missing.
   - Persist per-frame artifacts:
     - candidates.json: all raw proposals
     - candidates.npy (optional): packed arrays if large
     - filtered.json: proposals after filtering (min area, stability_score threshold, etc.)
3) Modified Algorithm 1 with SAM2 (run separately for each level):
   - Initialize SAM2 video predictor on the source image sequence.
   - For each frame t, compare current candidates with previously tracked masklets (IoU > κ, κ=0.6) to decide tracked vs. untracked.
   - For untracked bboxes, add as prompts and propagate masks across the sequence using SAM2 to produce masklets.
   - Merge per-iteration results into a final per-frame map of object_id → mask for that level.
4) Visualization & Storage:
   - Save a folder per object_id containing masked PNGs per frame.
   - Save summary stats (counts per level, per frame), and a manifest with model/config/ckpt and frame selection info.

Output Layout (under My3DIS)
- My3DIS/
  - Agent.md (this file)
  - algorithm1.ipynb (original; may import shared helpers)
  - run_pipeline.py (all-in-one; requires both Semantic-SAM and SAM2 in one env)
  - generate_candidates.py (Semantic-SAM only; use env: Semantic-SAM)
  - track_from_candidates.py (SAM2 only; use env: SAM2)
  - outputs/
    - <scene>/
      - manifest.json
      - logs/
      - demo/ (first 3 sampled frames)
        - level_2/{candidates,filtered,tracking,viz}
        - level_4/{candidates,filtered,tracking,viz}
        - level_6/{candidates,filtered,tracking,viz}
      - full/ (full selection, same structure as above)

Reproducibility Artifacts
- Per-level, per-frame files:
  - candidates.json: list of {frame_idx, bbox:[x,y,w,h], area, stability_score, level, meta}
  - filtered.json: same format after filtering
- Tracking outputs:
  - video_segments.npz or .pkl: {frame_idx: {obj_id: mask_bool_array}}
  - objects/<obj_id>/*.png: masked images
- Manifest (manifest.json): scene, range, levels, thresholds, model ckpt paths, code version (git commit when available).

Planned CLI (run_pipeline.py)
- Example (demo, 3 frames):
  - python My3DIS/run_pipeline.py \
    --data-path /media/public_dataset2/multiscan/<scene>/outputs/color \
    --levels 2,4,6 \
    --frames 1200:1600:20 \
    --sam-ckpt /media/Pluto/richkung/Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth \
    --sam2-cfg /media/Pluto/richkung/SAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2-ckpt /media/Pluto/richkung/SAM2/checkpoints/sam2.1_hiera_large.pt \
    --output My3DIS/outputs/<scene>/demo \
    --max-frames 3
- Then remove --max-frames to process the full selection into My3DIS/outputs/<scene>/full.

Two-stage Option (recommended with existing envs)
- Stage 1: Generate candidates (Semantic-SAM env)
  - conda run -n Semantic-SAM python My3DIS/generate_candidates.py \
    --data-path /media/public_dataset2/multiscan/<scene>/outputs/color \
    --levels 2,4,6 \
    --frames 1200:1600:20 \
    --sam-ckpt /media/Pluto/richkung/Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth \
    --output /media/Pluto/richkung/My3DIS/outputs/<scene>/demo \
    --max-frames 3
- Stage 2: Track with SAM2 (SAM2 env)
  - conda run -n SAM2 python My3DIS/track_from_candidates.py \
    --data-path /media/public_dataset2/multiscan/<scene>/outputs/color \
    --candidates-root /media/Pluto/richkung/My3DIS/outputs/<scene>/demo \
    --sam2-cfg /media/Pluto/richkung/SAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2-ckpt /media/Pluto/richkung/SAM2/checkpoints/sam2.1_hiera_large.pt \
    --output /media/Pluto/richkung/My3DIS/outputs/<scene>/demo

GitHub
- Repo: https://github.com/RICHKUNG/3DIS
- Push steps (from My3DIS/):
  - git init && git checkout -b main
  - git remote add origin https://github.com/RICHKUNG/3DIS.git
  - git add Agent.md run_pipeline.py Algorithm1_env.yml algorithm1.ipynb
  - git commit -m "Init: multi-level Semantic-SAM → SAM2 pipeline"
  - git push -u origin main

Notes & Assumptions
- Use Semantic-SAM to replace SAM in Algorithm 1; run the modified Algorithm 1 independently for each level in [2,4,6].
- MultiScan path is correct; do not write under /media/public_dataset2.
- Bounding boxes are in XYWH for candidate storage; convert to x1y1x2y2 and scale to SAM2 resolution before predictor prompts.
- IoU threshold κ defaults to 0.6; configurable later.

Status
- Plan agreed: levels [2,4,6], frames 1200:1600:20, demo on first 3 sampled frames.
- Decision: save both raw candidate bbox lists and filtered lists per level.
- Progress:
  - Added My3DIS/run_pipeline.py (multi-level Semantic-SAM → SAM2 tracking, per-level candidate persistence).
  - No code executed yet.
- Next actions:
  1) Create conda env from Algorithm1_env.yml and register kernel.
  2) Optional: Update notebook to import shared helpers (non-destructive) or copy required functions.
  3) Run demo (first 3 sampled frames), verify outputs and metrics.
  4) Run full selection.
  5) Push Agent.md and run_pipeline.py to GitHub repo https://github.com/RICHKUNG/3DIS.

Open Items
- Confirm the target scene(s) under MultiScan for the first run.
- Optional thresholds for filtering: min_area, stability_score (defaults can be tuned after the demo).


[Github repo](https://github.com/RICHKUNG/3DIS)
