export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# set start and end index (all are included), 27(0050), 243(0011), 304(0100)
n=9 # start
m=9 # end

# 31 ~ 45
# 46 ~ 60
# 61 ~ 70
# 71 ~ 80

# 81 ~ 105
# 106 ~ 130




DATASET_NAME="multiscan"
SAVE_OUTPUT_PLY_PTH=false
SAVE_EACH_PLY=false
# IS_GT=false

# MASK_TYPE="semanticSAM"
# MASK_TYPE="SAM2_max_overlap_visible_point"
# MASK_TYPE="SAM2_max_overlap_6_4_2"
# MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4"
# MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4_vis_new_pivot"
MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4_vis_orig_pivot"
# MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4_naive_pivot"
# MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4_search3d_segmentator"
# MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4_point_prompt"
# MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4_freq25"
# MASK_TYPE="SAM2_max_overlap_mask3d_sp_new_level4_overlap"


DATA_FOLDER=${DATA_FOLDER:-"./data/auto_generation/multiscan"}
DATASET_ROOT=${DATASET_ROOT:-"./data/multiscan"}
OUTPUT_BASE=${OUTPUT_BASE:-"./outputs/evaluation/${DATASET_NAME}_${MASK_TYPE}"}
EVALUATION_SCRIPT=${EVALUATION_SCRIPT:-"../Uni3DIS/run_evaluation.py"}
CUDA_DEVICE=${CUDA_DEVICE:-0}

SCENE_LIST=$(head -n $(($m)) "${DATASET_ROOT}/scans.txt" | tail -n $(($m - $n + 1)))


# # Randomly select n scenes from SCENE_LIST
# SELECTED_SCENES=$(echo "$SCENE_LIST" | shuf -n 10)

for SCENE_NAME in $SCENE_LIST; do
    SCENE_DIR="$DATASET_ROOT"

    echo "[SCENE] Processing scene: $SCENE_NAME"
    echo "[SCENE] Scene directory: $SCENE_DIR"

    # INPUT
    SCENE_PLY_PATH="$DATASET_ROOT"

    OUTPUT_FOLDER="${OUTPUT_BASE}/${SCENE_NAME}"

    # OUTPUT
    # OUTPUT_FOLDER="/path/to/custom/output/top${topk}_pair${topk_per_image}"

    # Check if the output folder exists, if not, create it
    if [ ! -d "$OUTPUT_FOLDER" ]; then
        mkdir -p "$OUTPUT_FOLDER"
        echo "Created directory: $OUTPUT_FOLDER"
    fi    

    # echo "[OUTPUT] OUTPUT directory: $OUTPUT_FOLDER"

    # run single scene inference
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    python "$EVALUATION_SCRIPT" \
    --dataset_name=${DATASET_NAME} \
    --input_rgb_dataset_path=${SCENE_DIR} \
    --scene_name=${SCENE_NAME} \
    --path_to_ply=${SCENE_PLY_PATH} \
    --output_path=${OUTPUT_FOLDER} \
    --data=${DATA_FOLDER}

done
