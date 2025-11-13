#!/bin/bash
# Full GT SigLIP similarity test with optimized pipeline

cd /media/Pluto/richkung/My3DIS

export PYTHONPATH=src

echo "Running FULL GT test with optimized pipeline..."
echo "================================================"

time python scripts/check_siglip_gt_similarity.py \
  --scene-path /media/public_dataset2/multiscan/scene_00005_00 \
  --annotation-file data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations/scene_00005_00_obj_part_inst.txt \
  --siglip-model /media/Pluto/richkung/3DprojToSiglip/models/siglip-so400m-patch14-384 \
  --local-files-only \
  --device cuda:0 \
  --min-points 150 \
  --topk-views 1 \
  --batch-size 32 \
  --output-json outputs/gt_siglip_check/scene_00005_00/similarity_results_full.json

echo ""
echo "================================================"
echo "Test completed!"
