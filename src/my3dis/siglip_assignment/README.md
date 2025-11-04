# SigLIP Feature Assignment Module

This module implements the second stage of the 3D projection pipeline, which assigns vision-language features from SigLIP to 3D proposals for open-vocabulary querying.

## Overview

**Input**: 3D proposals from aggregation stage (`proposal_data_level{2,4,6}.npz`)
**Output**: Proposals with SigLIP features (`siglip_features_level{2,4,6}.npz`)

### Pipeline Flow

```
proposal_data_level{L}.npz (from aggregation)
    ↓
For each proposal:
  - Find best K views (frames)
  - Extract multi-scale crops from each view
  - Encode with SigLIP vision encoder
  - Aggregate features across views
    ↓
(Optional) Assign labels via text queries
    ↓
Export proposals with features (NPZ)
```

## Module Structure

```
siglip_assignment/
├── __init__.py                 # Package exports
├── siglip_pipeline.py          # Main pipeline orchestration
├── feature_extraction.py       # SigLIP feature extraction
├── assignment_utils.py         # Open-vocabulary inference
└── README.md                   # This file
```

## Key Components

### 1. SigLIPAssignmentPipeline

Main orchestrator that:
- Loads 3D proposals from aggregation output
- Extracts SigLIP features from 2D views
- Aggregates features across frames
- (Optional) Assigns semantic labels
- Exports results

**Key Parameters**:
- `model_name` (default: 'google/siglip-so400m-patch14-384') - HuggingFace model
- `batch_size` (default: 32) - Proposals processed in parallel
- `topk_frames` (default: 5) - Best views per proposal

### 2. SigLIPFeatureExtractor

Handles feature extraction:
- Loads SigLIP model from HuggingFace
- Multi-scale cropping for better coverage
- Batched processing for efficiency
- L2-normalized features

**Multi-scale Strategy**:
1. Crop proposal bounding box
2. Expand by 20% (kexp=0.2)
3. Crop again (3 scales total)
4. Average features across scales

### 3. Assignment Utilities

**Open-Vocabulary Inference Methods**:
- `assign_labels()` - Standard object-level classification
- `assign_part_labels()` - Object-part hierarchical fusion
- `compute_mask_scores()` - Per-point to mask aggregation

**Similarity Scoring**:
- Temperature scaling (default: 300.0)
- Softmax normalization
- Top-K or best-match selection

## Usage

### Standalone CLI

```bash
PYTHONPATH=src python -m my3dis.run_siglip_assignment \
  --proposal-dir outputs/aggregation/scene_00065_00 \
  --color-frames-dir /path/to/scene/color \
  --depth-maps-dir /path/to/scene/depth \
  --output-dir outputs/siglip/scene_00065_00 \
  --levels 2 4 6 \
  --model google/siglip-so400m-patch14-384 \
  --batch-size 32 \
  --topk-frames 5 \
  --labels chair table sofa lamp \
  --device cuda:0
```

### Workflow Integration (YAML)

```yaml
stages:
  siglip_assignment:
    enabled: true

    # Model configuration
    model_name: google/siglip-so400m-patch14-384
    batch_size: 32
    topk_frames: 5
    use_multiscale: true

    # Inference configuration
    scale_semantic_score: 300.0
    duplicate_topk: 600
    allow_duplicates: true

    device: cuda:0
```

### Python API

```python
from my3dis.siglip_assignment import SigLIPAssignmentPipeline

pipeline = SigLIPAssignmentPipeline(
    proposal_dir='aggregation_output/',
    color_frames_dir='color/',
    depth_maps_dir='depth/',
    model_name='google/siglip-so400m-patch14-384',
    batch_size=32,
    topk_frames=5,
)

# Extract features only
results = pipeline.run(
    output_dir='siglip_output/',
    levels=[2, 4, 6],
)

# Extract features + assign labels
results = pipeline.run(
    output_dir='siglip_output/',
    levels=[2, 4, 6],
    object_labels=['chair', 'table', 'sofa', 'lamp'],
)
```

## Output Format

### siglip_features_level{L}.npz

Each NPZ file contains:

```python
{
    # From aggregation stage
    'masks': np.ndarray,        # [N_points, N_proposals] binary masks
    'obj_ids': np.ndarray,      # [N_proposals] object IDs
    'levels': np.ndarray,       # [N_proposals] level numbers
    'parent_ids': np.ndarray,   # [N_proposals] parent IDs
    'volumes': np.ndarray,      # [N_proposals] volumes (m³)
    'centroids': np.ndarray,    # [N_proposals, 3] centroids
    'bboxes': np.ndarray,       # [N_proposals, 6] bboxes
    'points': np.ndarray,       # [N_points, 3] coordinates
    'colors': np.ndarray,       # [N_points, 3] colors

    # New: SigLIP features
    'features': np.ndarray,     # [N_proposals, 1152] L2-normalized features

    # Optional: if labels provided
    'pred_classes': np.ndarray, # [N_proposals] predicted class indices
    'pred_scores': np.ndarray,  # [N_proposals] confidence scores
}
```

### siglip_metadata.json

```json
{
  "proposal_dir": "aggregation_output/",
  "model": "google/siglip-so400m-patch14-384",
  "batch_size": 32,
  "topk_frames": 5,
  "outputs": {
    "2": "siglip_features_level2.npz",
    "4": "siglip_features_level4.npz",
    "6": "siglip_features_level6.npz"
  }
}
```

## Feature Extraction Details

### Multi-Scale Cropping

For each proposal in each frame:

```python
# Initial crop: proposal bounding box
crop_1 = image[y1:y2, x1:x2]

# Scale 2: Expand by 20%
y1 -= 0.2 * (y2 - y1)
x1 -= 0.2 * (x2 - x1)
crop_2 = image[y1:y2, x1:x2]

# Scale 3: Expand again
crop_3 = ...

# Encode and average
features = siglip_encoder([crop_1, crop_2, crop_3]).mean(dim=0)
```

### View Selection

Currently simplified (top-K by score). Future improvements:
- Camera pose-based selection
- Visibility-based ranking
- Coverage-based sampling

### Feature Aggregation

Across multiple frames:

```python
frame_features = [extract(frame_i) for frame_i in best_frames]
aggregated = torch.stack(frame_features).mean(dim=0)
aggregated = F.normalize(aggregated, dim=-1)  # L2 normalize
```

## Open-Vocabulary Inference

### Basic Object Classification

```python
from my3dis.siglip_assignment.assignment_utils import assign_labels

# Features: [N_proposals, 1152]
# Text features: [N_labels, 1152]
results = assign_labels(
    features=proposal_features,
    text_features=text_features,
    scale_factor=300.0,
    duplicate=False,  # Best match per proposal
)

# results['pred_classes']: [N_proposals] label indices
# results['pred_scores']: [N_proposals] confidence scores
```

### Top-K Retrieval (Duplicate Mode)

```python
results = assign_labels(
    features=proposal_features,
    text_features=text_features,
    duplicate=True,  # Allow multiple instances per label
    topk=600,
)

# Returns top 600 (proposal, label) pairs sorted by score
```

### Object-Part Fusion

```python
from my3dis.siglip_assignment.assignment_utils import assign_part_labels

results = assign_part_labels(
    features_object=object_features,
    features_part=part_features,
    text_features_object=text_features_obj,
    text_features_part=text_features_part,
    part_to_object_mapping={0: 0, 1: 0, 2: 1},  # part_idx -> obj_idx
)

# Fuses object-level and part-level scores
```

## Supported Models

### SigLIP Variants

- `google/siglip-base-patch16-224` (768-dim, fastest)
- `google/siglip-large-patch16-384` (1024-dim)
- **`google/siglip-so400m-patch14-384`** (1152-dim, recommended)

### Switching Models

```yaml
stages:
  siglip_assignment:
    model_name: google/siglip-base-patch16-224  # Faster, lower accuracy
```

Or via CLI:

```bash
python -m my3dis.run_siglip_assignment \
  --model google/siglip-base-patch16-224 \
  ...
```

## Performance Optimization

### Batching

Process multiple proposals together:

```yaml
batch_size: 64  # Higher = faster but more memory
```

### Frame Selection

Reduce frames per proposal:

```yaml
topk_frames: 3  # Lower = faster but less accurate
```

### Multi-Scale Toggle

Disable for speed:

```yaml
use_multiscale: false  # Single scale only
```

## Dependencies

**Required**:
- torch
- transformers (HuggingFace)
- PIL
- numpy
- tqdm

Install:

```bash
pip install torch transformers pillow numpy tqdm
```

## Integration Example

### Full Pipeline (Aggregation → SigLIP)

```bash
# Step 1: Aggregation
PYTHONPATH=src python -m my3dis.run_aggregation \
  --family-tree relations/family_tree.json \
  --pointcloud pointcloud.ply \
  --depth-maps-dir depth/ \
  --color-frames-dir color/ \
  --output-dir output/aggregation/

# Step 2: SigLIP Assignment
PYTHONPATH=src python -m my3dis.run_siglip_assignment \
  --proposal-dir output/aggregation/ \
  --color-frames-dir color/ \
  --depth-maps-dir depth/ \
  --output-dir output/siglip/ \
  --labels chair table sofa lamp desk
```

### Querying Results

```python
import numpy as np

# Load results
data = np.load('output/siglip/siglip_features_level2.npz')

features = data['features']  # [N, 1152]
pred_classes = data['pred_classes']  # [N]
pred_scores = data['pred_scores']  # [N]

labels = ['chair', 'table', 'sofa', 'lamp', 'desk']

# Find all chairs
chair_idx = 0
chair_proposals = np.where(pred_classes == chair_idx)[0]
print(f"Found {len(chair_proposals)} chair proposals")

# Get high-confidence predictions
confident = pred_scores > 0.5
print(f"{confident.sum()} high-confidence predictions")
```

## Troubleshooting

### Out of Memory

Reduce batch size or frame count:

```yaml
batch_size: 16
topk_frames: 3
```

### Model Download Fails

Pre-download model:

```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('google/siglip-so400m-patch14-384')
processor = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')
```

### Low Accuracy

- Increase `topk_frames` for more views
- Enable `use_multiscale`
- Try larger model (siglip-so400m)

## Next Steps

After SigLIP assignment, use features for:

1. **Open-vocabulary query**: "Find all chairs"
2. **Similarity search**: "Find objects similar to this"
3. **Hierarchical refinement**: Combine with family tree
4. **Downstream tasks**: Instance segmentation, scene understanding

## Related Modules

- `../aggregation/` - Generates 3D proposals
- `../family_tree_query.py` - Query family relationships
- `../inference/` - Full open-vocabulary pipeline (under development)
