# 3D Mask Aggregation Module

This module implements the first stage of the 3D projection pipeline, which projects 2D instance masks from SAM2 tracking to 3D point clouds and generates 3D proposals.

## Overview

**Input**: `family_tree.json` from SAM2 tracking stage
**Output**: 3D proposals in NPZ format (`proposal_data_level{2,4,6}.npz`)

### Pipeline Flow

```
family_tree.json (SAM2 output)
    ↓
Load objects by level
    ↓
For each object:
  - Load 2D masks from all frames
  - Project to 3D using depth maps & camera params
  - Aggregate across frames (union/intersection/majority)
    ↓
Filter by volume threshold
    ↓
Deduplicate by 3D IoU
    ↓
Export 3D proposals (NPZ)
```

## Module Structure

```
aggregation/
├── __init__.py                 # Package exports
├── aggregation_pipeline.py     # Main pipeline orchestration
├── mask_projection.py          # 2D→3D projection logic
├── aggregation_utils.py        # Utility functions
└── README.md                   # This file
```

## Key Components

### 1. AggregationPipeline

Main orchestrator that:
- Loads family tree from SAM2 output
- Processes objects level by level
- Coordinates projection and aggregation
- Applies filtering and deduplication
- Exports results

**Key Parameters**:
- `iou_threshold` (default: 0.6) - IoU threshold for 3D deduplication
- `min_volume` (default: 0.001 m³) - Minimum volume filter
- `aggregation_method` (default: 'union') - How to aggregate masks across frames

### 2. MaskProjector

Handles 2D to 3D projection:
- Projects 3D points to 2D using camera parameters
- Checks visibility using depth maps
- Filters by depth consistency (default: 0.05m threshold)

**Depth Map Formats Supported**:
- PNG (assumed mm → meters conversion)
- EXR (requires OpenEXR library)

### 3. Aggregation Utilities

**Aggregation Methods**:
- `union`: Any frame where point is visible
- `intersection`: Only points visible in ALL frames
- `majority`: Points visible in >50% of frames

**Geometric Properties Computed**:
- Centroid (center of mass)
- Bounding box (axis-aligned)
- Volume (approximate from bbox)
- Surface area (approximate)

## Usage

### Standalone CLI

```bash
PYTHONPATH=src python -m my3dis.run_aggregation \
  --family-tree outputs/experiments/scene_00065_00/run_001/relations/family_tree.json \
  --pointcloud /path/to/scene/pointcloud.ply \
  --depth-maps-dir /path/to/scene/depth \
  --color-frames-dir /path/to/scene/color \
  --output-dir outputs/aggregation/scene_00065_00 \
  --levels 2 4 6 \
  --iou-threshold 0.6 \
  --min-volume 0.001 \
  --device cuda:0
```

### Workflow Integration (YAML)

```yaml
stages:
  aggregation:
    enabled: true
    pointcloud_path: /path/to/scene/pointcloud.ply
    depth_maps_dir: /path/to/scene/depth
    camera_params_path: /path/to/camera_params.json  # Optional

    # Hyperparameters
    iou_threshold: 0.6
    min_volume: 0.001  # m³
    vis_depth_threshold: 0.05  # meters
    aggregation_method: union

    device: cuda:0
```

### Python API

```python
from my3dis.aggregation import AggregationPipeline

pipeline = AggregationPipeline(
    family_tree_path='relations/family_tree.json',
    pointcloud_path='pointcloud.ply',
    depth_maps_dir='depth/',
    color_frames_dir='color/',
    camera_params_path='camera_params.json',
    iou_threshold=0.6,
    min_volume=0.001,
)

results = pipeline.run(
    output_dir='aggregation_output/',
    levels=[2, 4, 6],
)

# results = {'2': 'proposal_data_level2.npz', ...}
```

## Output Format

### proposal_data_level{L}.npz

Each NPZ file contains:

```python
{
    'masks': np.ndarray,        # [N_points, N_proposals] binary masks
    'obj_ids': np.ndarray,      # [N_proposals] object IDs from family tree
    'levels': np.ndarray,       # [N_proposals] level numbers (all same)
    'parent_ids': np.ndarray,   # [N_proposals] parent IDs (-1 if root)
    'volumes': np.ndarray,      # [N_proposals] volumes (m³)
    'centroids': np.ndarray,    # [N_proposals, 3] centroids
    'bboxes': np.ndarray,       # [N_proposals, 6] bboxes (min_xyz, max_xyz)
    'points': np.ndarray,       # [N_points, 3] point cloud coordinates
    'colors': np.ndarray,       # [N_points, 3] point cloud colors
}
```

### aggregation_metadata.json

```json
{
  "family_tree": "path/to/family_tree.json",
  "pointcloud": "path/to/pointcloud.ply",
  "iou_threshold": 0.6,
  "min_volume": 0.001,
  "outputs": {
    "2": "proposal_data_level2.npz",
    "4": "proposal_data_level4.npz",
    "6": "proposal_data_level6.npz"
  },
  "statistics": {
    "total_objects": 150,
    "total_families": 45
  }
}
```

## Integration with Family Tree

The aggregation stage directly consumes `family_tree.json` from SAM2 tracking:

```python
from my3dis.family_tree_query import FamilyTreeQuery

# Load family tree
query = FamilyTreeQuery('family_tree.json')

# Get objects at level 2
obj_ids = query.search_objects_by_level(2)

# For each object, load masks
for obj_id in obj_ids:
    frames = query.get_object_frames(obj_id)
    masks = query.load_masks_batch(obj_id, frames)
    # Project to 3D...
```

## Camera Parameters Format

If provided, `camera_params.json` should contain:

```json
{
  "frames": [
    {
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "R": [[...], [...], [...]],
      "t": [tx, ty, tz]
    }
  ]
}
```

Or global parameters:

```json
{
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
}
```

## Dependencies

**Required**:
- torch
- numpy
- PIL

**Optional**:
- plyfile (for .ply point cloud reading)
- OpenEXR (for .exr depth maps)

Install optional dependencies:

```bash
pip install plyfile OpenEXR
```

## Performance Considerations

- **Memory**: Masks are processed level by level to reduce peak memory
- **Caching**: Depth maps are cached during processing
- **Batching**: Can be parallelized across levels (not implemented yet)

## Troubleshooting

### "Depth map not found"

Ensure depth maps follow naming convention:
- `depth_XXXXXX.png`
- `depth_XXXXXX.exr`
- `XXXXXX_depth.png`

### "No valid projections"

Check:
1. Camera parameters are correctly formatted
2. Depth maps have correct scale (PNG should be in mm)
3. Visibility threshold is not too strict

### "Low volume after filtering"

Adjust `min_volume` threshold:
- Default: 0.001 m³ (1 liter)
- For small objects: 0.0001 m³
- For large objects only: 0.01 m³

## Next Stage

Output proposals are consumed by the **SigLIP Assignment** stage:

```
proposal_data_level{L}.npz → SigLIP Feature Extraction → siglip_features_level{L}.npz
```

See `../siglip_assignment/README.md` for details.
