"""
3D-to-2D mask projection for GT proposal visualization.

This module handles projecting 3D point cloud masks to 2D RGB frames
using camera intrinsics and extrinsics (pose).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Mask3Dto2DProjector:
    """
    Projects 3D point cloud masks to 2D image masks using camera parameters.

    This is the reverse operation of MaskProjector (which projects 2D to 3D).
    """

    def __init__(
        self,
        intrinsics_path: str,
        pose_dir: str,
        depth_dir: Optional[str] = None,
        vis_depth_threshold: float = 0.05,
    ):
        """
        Initialize 3D-to-2D mask projector.

        Args:
            intrinsics_path: Path to camera intrinsics file (4x4 matrix)
            pose_dir: Directory containing pose files (frame_idx.txt)
            depth_dir: Optional directory containing depth maps for visibility check
            vis_depth_threshold: Visibility depth threshold (meters)
        """
        self.intrinsics_path = Path(intrinsics_path)
        self.pose_dir = Path(pose_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.vis_depth_threshold = vis_depth_threshold

        # Load intrinsics
        self.K = self._load_intrinsics()

        # Cache for poses and depth maps
        self._pose_cache: Dict[int, np.ndarray] = {}
        self._depth_cache: Dict[int, np.ndarray] = {}

        logger.info(f"Initialized Mask3Dto2DProjector")
        logger.info(f"  Intrinsics: {intrinsics_path}")
        logger.info(f"  Pose dir: {pose_dir}")
        if self.depth_dir:
            logger.info(f"  Depth dir: {depth_dir} (visibility check enabled)")

    def _load_intrinsics(self) -> np.ndarray:
        """Load camera intrinsics matrix."""
        K = np.loadtxt(self.intrinsics_path)
        if K.shape == (4, 4):
            # Extract 3x3 intrinsics from 4x4 matrix
            K = K[:3, :3]
        return K

    def _load_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load camera pose (world-to-camera transform) for frame."""
        if frame_idx in self._pose_cache:
            return self._pose_cache[frame_idx]

        pose_path = self.pose_dir / f"{frame_idx}.txt"
        if not pose_path.exists():
            logger.debug(f"Pose file not found for frame {frame_idx}")
            return None

        pose = np.loadtxt(pose_path)  # 4x4 camera-to-world
        if pose.shape != (4, 4):
            logger.warning(f"Invalid pose shape: {pose.shape}")
            return None

        # Convert camera-to-world to world-to-camera
        pose_inv = np.linalg.inv(pose)

        self._pose_cache[frame_idx] = pose_inv
        return pose_inv

    def _load_depth_map(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load depth map for visibility check."""
        if not self.depth_dir:
            return None

        if frame_idx in self._depth_cache:
            return self._depth_cache[frame_idx]

        # Try common naming conventions
        possible_names = [
            f"{frame_idx}.png",
            f"{frame_idx:06d}.png",
            f"depth_{frame_idx}.png",
        ]

        for name in possible_names:
            depth_path = self.depth_dir / name
            if depth_path.exists():
                depth_img = Image.open(depth_path)
                depth = np.array(depth_img).astype(np.float32) / 1000.0  # mm to meters
                self._depth_cache[frame_idx] = depth
                return depth

        logger.debug(f"Depth map not found for frame {frame_idx}")
        return None

    def project_3d_mask_to_2d(
        self,
        points: np.ndarray,
        mask_3d: np.ndarray,
        frame_idx: int,
        image_size: Tuple[int, int],
        check_visibility: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Project 3D point cloud mask to 2D image mask.

        Args:
            points: Point cloud [N_points, 3]
            mask_3d: Binary mask [N_points]
            frame_idx: Frame index
            image_size: (height, width) of target image
            check_visibility: Whether to check depth visibility

        Returns:
            Binary 2D mask [H, W] or None if projection fails
        """
        # Load pose
        pose = self._load_pose(frame_idx)
        if pose is None:
            return None

        # Filter points by mask
        masked_points = points[mask_3d]
        if len(masked_points) == 0:
            return None

        # Transform to camera coordinates
        # points_cam = R @ points + t
        # where [R | t] = pose (world-to-camera)
        #       [0 | 1]
        points_homo = np.concatenate([masked_points, np.ones((len(masked_points), 1))], axis=1)
        points_cam_homo = (pose @ points_homo.T).T
        points_cam = points_cam_homo[:, :3]

        # Filter points behind camera
        valid_depth = points_cam[:, 2] > 0.1  # At least 0.1m in front
        if not valid_depth.any():
            return None

        points_cam = points_cam[valid_depth]

        # Project to image plane
        points_2d_homo = (self.K @ points_cam.T).T
        depths = points_2d_homo[:, 2]
        points_2d = points_2d_homo[:, :2] / depths[:, None]

        # Round to pixel coordinates
        pixels = np.round(points_2d).astype(int)

        # Filter points outside image bounds
        H, W = image_size
        valid_pixels = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < W) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < H)
        )

        if not valid_pixels.any():
            return None

        pixels = pixels[valid_pixels]
        pixel_depths = depths[valid_pixels]

        # Visibility check using depth map
        if check_visibility and self.depth_dir:
            depth_map = self._load_depth_map(frame_idx)
            if depth_map is not None:
                # Filter occluded points
                visible_mask = np.zeros(len(pixels), dtype=bool)
                for i, (u, v) in enumerate(pixels):
                    depth_at_pixel = depth_map[v, u]
                    if abs(pixel_depths[i] - depth_at_pixel) <= self.vis_depth_threshold:
                        visible_mask[i] = True

                pixels = pixels[visible_mask]

        if len(pixels) == 0:
            return None

        # Create 2D mask
        mask_2d = np.zeros((H, W), dtype=bool)
        mask_2d[pixels[:, 1], pixels[:, 0]] = True

        return mask_2d

    def get_visible_frames(
        self,
        points: np.ndarray,
        mask_3d: np.ndarray,
        frame_indices: List[int],
        image_size: Tuple[int, int],
        min_visible_points: int = 50,
    ) -> List[int]:
        """
        Get list of frames where the 3D mask is visible.

        Args:
            points: Point cloud [N_points, 3]
            mask_3d: Binary mask [N_points]
            frame_indices: List of frame indices to check
            image_size: (height, width) of images
            min_visible_points: Minimum number of visible points

        Returns:
            List of frame indices where mask is visible
        """
        visible_frames = []

        for frame_idx in frame_indices:
            mask_2d = self.project_3d_mask_to_2d(
                points, mask_3d, frame_idx, image_size, check_visibility=True
            )

            if mask_2d is not None and mask_2d.sum() >= min_visible_points:
                visible_frames.append(frame_idx)

        return visible_frames


def main():
    """CLI test."""
    import argparse

    parser = argparse.ArgumentParser(description='Test 3D-to-2D mask projection')
    parser.add_argument('--intrinsics', required=True, help='Path to intrinsics.txt')
    parser.add_argument('--pose-dir', required=True, help='Directory with pose files')
    parser.add_argument('--depth-dir', help='Directory with depth maps (optional)')
    parser.add_argument('--pointcloud', required=True, help='Path to point cloud .ply')
    parser.add_argument('--test-frames', nargs='+', type=int, default=[0, 100, 200],
                        help='Frame indices to test')

    args = parser.parse_args()

    # Load point cloud
    from plyfile import PlyData
    ply = PlyData.read(args.pointcloud)
    vertex = ply['vertex']
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Create dummy mask (first 1000 points)
    mask_3d = np.zeros(len(points), dtype=bool)
    mask_3d[:1000] = True

    # Initialize projector
    projector = Mask3Dto2DProjector(
        intrinsics_path=args.intrinsics,
        pose_dir=args.pose_dir,
        depth_dir=args.depth_dir,
    )

    # Test projection
    image_size = (1080, 1920)  # Assume HD resolution
    for frame_idx in args.test_frames:
        mask_2d = projector.project_3d_mask_to_2d(
            points, mask_3d, frame_idx, image_size, check_visibility=True
        )

        if mask_2d is not None:
            print(f"Frame {frame_idx}: {mask_2d.sum()} visible pixels")
        else:
            print(f"Frame {frame_idx}: No visible pixels")


if __name__ == '__main__':
    main()
