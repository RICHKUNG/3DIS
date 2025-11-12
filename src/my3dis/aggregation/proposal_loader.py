"""
Proposal loader - loads proposals from My3DIS tracking outputs without 3D projection.

This is a lightweight alternative to AggregationPipeline for cases where:
- 3D point cloud is not available
- Depth maps are not available
- Only 2D mask aggregation is needed (e.g., for evaluation)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from my3dis.tracking import load_legacy_video_segments, load_object_segments_manifest
from my3dis.mask.encoding import unpack_binary_mask
from my3dis.family_tree_query import FamilyTreeQuery

logger = logging.getLogger(__name__)


class ProposalLoader:
    """
    Load proposals from My3DIS tracking outputs.

    This class handles loading 2D masks from tracking outputs and aggregating
    them across frames. Unlike AggregationPipeline, it does not perform 3D
    projection and works directly with 2D mask data.
    """

    def __init__(self, exp_dir: str, dataset_root: str):
        """
        Args:
            exp_dir: Path to experiment directory (e.g., outputs/experiments/scene_XX/run_YY)
            dataset_root: MultiScan dataset root directory
        """
        self.exp_dir = Path(exp_dir)
        self.dataset_root = Path(dataset_root)

        if not self.exp_dir.exists():
            raise ValueError(f"Experiment directory not found: {exp_dir}")

        # Load family tree if available
        family_tree_path = self.exp_dir / "relations" / "family_tree.json"
        if family_tree_path.exists():
            self.family_query = FamilyTreeQuery(str(family_tree_path))
            logger.info(f"Loaded family tree from {family_tree_path}")
        else:
            self.family_query = None
            logger.warning(f"Family tree not found: {family_tree_path}")

    def get_scene_name(self) -> str:
        """Extract scene name from experiment directory."""
        # Check if directory name looks like a scene (scene_XXXXX_XX format)
        dir_name = self.exp_dir.name
        if dir_name.startswith('scene_'):
            return dir_name
        # Otherwise assume path structure: .../experiments/{scene_name}/{run_name}
        return self.exp_dir.parent.name

    def load_proposals(
        self,
        levels: List[str] = ['L2', 'L4', 'L6'],
        aggregation_method: str = 'union'
    ) -> Tuple[Dict[str, List[Dict]], int, str]:
        """
        Load proposals from My3DIS tracking outputs.

        Args:
            levels: Levels to load (e.g., ['L2', 'L4', 'L6'])
            aggregation_method: Method to aggregate masks across frames
                               ('union', 'intersection', 'majority')

        Returns:
            Tuple of (proposals_by_level, num_points, scene_name)
            - proposals_by_level: {level: [{'object_id': int, 'mask': np.ndarray, 'frames': list, ...}, ...]}
            - num_points: Total number of points (H * W for 2D masks)
            - scene_name: Scene identifier
        """
        scene_name = self.get_scene_name()
        proposals_by_level = {}
        num_points = None

        for level in levels:
            logger.info(f"Loading proposals for {level}...")

            # Find level directory
            level_dir = self._find_level_dir(level)
            if level_dir is None:
                logger.warning(f"Level directory not found for {level}")
                continue

            # Find tracking files
            tracking_files = self._find_tracking_files(level_dir)
            if tracking_files is None:
                logger.warning(f"Missing tracking files for {level}")
                continue

            video_npz, object_npz = tracking_files
            logger.info(f"Loading {level} from {object_npz.name} + {video_npz.name}")

            try:
                # Load tracking data
                proposals, level_num_points = self._load_level_proposals(
                    video_npz, object_npz, level, aggregation_method
                )

                # Set num_points from first level
                if num_points is None:
                    num_points = level_num_points
                    logger.info(f"Determined num_points={num_points}")

                proposals_by_level[level] = proposals
                logger.info(f"Loaded {len(proposals)} proposals from {level}")

            except Exception as e:
                logger.error(f"Error loading {level}: {e}", exc_info=True)
                continue

        if num_points is None:
            raise ValueError("Could not determine number of points from tracking outputs")

        return proposals_by_level, num_points, scene_name

    def _find_level_dir(self, level: str) -> Optional[Path]:
        """Find level directory for given level name."""
        # Try both formats: level_L2 and level_2
        level_num = level[1:] if level.startswith('L') else level

        candidates = [
            self.exp_dir / f"level_{level_num}",
            self.exp_dir / f"level_{level}"
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _find_tracking_files(self, level_dir: Path) -> Optional[Tuple[Path, Path]]:
        """Find video_segments and object_segments files."""
        # Try multiple locations: level_X/tracking/ or level_X/ directly
        search_dirs = [level_dir / "tracking", level_dir]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            video_files = list(search_dir.glob("video_segments*.npz"))
            object_files = list(search_dir.glob("object_segments*.npz"))

            if video_files and object_files:
                return sorted(video_files)[0], sorted(object_files)[0]

        return None

    def _load_level_proposals(
        self,
        video_npz: Path,
        object_npz: Path,
        level: str,
        aggregation_method: str
    ) -> Tuple[List[Dict], int]:
        """
        Load proposals for a single level.

        Returns:
            Tuple of (proposals, num_points)
        """
        # Load using My3DIS utilities
        video_frames, video_manifest = load_legacy_video_segments(
            str(video_npz),
            unpack_masks=False
        )
        object_map = load_object_segments_manifest(str(object_npz))

        proposals = []
        num_points = None

        for obj_id in sorted(object_map.keys()):
            frame_indices = list(object_map[obj_id].keys())

            if len(frame_indices) == 0:
                continue

            # Get mask shape from first frame
            if frame_indices[0] in video_frames:
                frame_data = video_frames[frame_indices[0]]
                if obj_id in frame_data:
                    sample_mask = unpack_binary_mask(frame_data[obj_id])
                    H, W = sample_mask.shape

                    # Set num_points if not yet determined
                    if num_points is None:
                        num_points = H * W

                    # Aggregate masks across frames
                    aggregated_mask = self._aggregate_masks(
                        obj_id, frame_indices, video_frames, (H, W), aggregation_method
                    )

                    # Get additional metadata from family tree if available
                    metadata = {}
                    if self.family_query is not None:
                        obj_info = self.family_query.get_object_info(obj_id)
                        if obj_info:
                            metadata = {
                                'parent_id': obj_info.get('parent_id'),
                                'children': obj_info.get('children', []),
                                'num_frames_in_tree': len(obj_info.get('frames', []))
                            }

                    proposals.append({
                        'object_id': int(obj_id),
                        'mask': aggregated_mask.flatten(),  # Flatten to 1D
                        'frames': frame_indices,
                        'level': level,
                        'num_frames': len(frame_indices),
                        **metadata
                    })

        return proposals, num_points

    def _aggregate_masks(
        self,
        obj_id: int,
        frame_indices: List[int],
        video_frames: Dict,
        shape: Tuple[int, int],
        method: str
    ) -> np.ndarray:
        """
        Aggregate masks across frames.

        Args:
            obj_id: Object ID
            frame_indices: List of frame indices
            video_frames: Frame data dict
            shape: Mask shape (H, W)
            method: Aggregation method ('union', 'intersection', 'majority')

        Returns:
            Aggregated mask [H, W]
        """
        H, W = shape
        masks = []

        for frame_idx in frame_indices:
            if frame_idx in video_frames:
                frame_data = video_frames[frame_idx]
                if obj_id in frame_data:
                    mask = unpack_binary_mask(frame_data[obj_id])
                    masks.append(mask)

        if len(masks) == 0:
            return np.zeros((H, W), dtype=bool)

        if len(masks) == 1:
            return masks[0]

        # Stack masks and aggregate
        masks_stack = np.stack(masks, axis=0)

        if method == 'union':
            return masks_stack.any(axis=0)
        elif method == 'intersection':
            return masks_stack.all(axis=0)
        elif method == 'majority':
            votes = masks_stack.sum(axis=0)
            return votes >= (len(masks) / 2)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


def load_proposals_from_experiment(
    exp_dir: str,
    dataset_root: str,
    levels: List[str] = ['L2', 'L4', 'L6'],
    aggregation_method: str = 'union'
) -> Tuple[Dict[str, List[Dict]], int, str]:
    """
    Convenience function to load proposals from experiment directory.

    Args:
        exp_dir: Path to experiment directory
        dataset_root: MultiScan dataset root directory
        levels: Levels to load
        aggregation_method: Method to aggregate masks across frames

    Returns:
        Tuple of (proposals_by_level, num_points, scene_name)
    """
    loader = ProposalLoader(exp_dir, dataset_root)
    return loader.load_proposals(levels, aggregation_method)
