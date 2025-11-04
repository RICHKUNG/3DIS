"""
2D mask projection to 3D point cloud.

Handles camera projection, depth alignment, and visibility checks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class MaskProjector:
    """
    Projects 2D masks to 3D point cloud using depth maps and camera parameters.
    """

    def __init__(
        self,
        pointcloud_path: str,
        depth_maps_dir: str,
        camera_params_path: Optional[str] = None,
        device: str = 'cuda:0',
        vis_depth_threshold: float = 0.05,
    ):
        """
        Initialize mask projector.

        Args:
            pointcloud_path: Path to point cloud (.ply or .pth)
            depth_maps_dir: Directory containing depth maps
            camera_params_path: Path to camera parameters JSON
            device: Torch device
            vis_depth_threshold: Visibility depth threshold (meters)
        """
        self.pointcloud_path = Path(pointcloud_path)
        self.depth_maps_dir = Path(depth_maps_dir)
        self.device = torch.device(device)
        self.vis_depth_threshold = vis_depth_threshold

        # Load point cloud
        self.points, self.colors = self._load_pointcloud()

        # Load camera parameters
        if camera_params_path:
            self.camera_params = self._load_camera_params(camera_params_path)
        else:
            self.camera_params = None

        # Cache for depth maps
        self._depth_cache: Dict[int, torch.Tensor] = {}

        logger.info(f"Initialized MaskProjector")
        logger.info(f"  Point cloud: {pointcloud_path} ({len(self.points)} points)")
        logger.info(f"  Depth maps: {depth_maps_dir}")

    def _load_pointcloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load point cloud from file."""
        if self.pointcloud_path.suffix == '.pth':
            data = torch.load(self.pointcloud_path, map_location='cpu')
            points = data['points'].numpy() if isinstance(data['points'], torch.Tensor) else data['points']
            colors = data.get('colors', np.ones_like(points) * 255)
            if isinstance(colors, torch.Tensor):
                colors = colors.numpy()
        elif self.pointcloud_path.suffix == '.ply':
            # Simple PLY parser (can be replaced with plyfile library)
            points, colors = self._load_ply(self.pointcloud_path)
        else:
            raise ValueError(f"Unsupported point cloud format: {self.pointcloud_path.suffix}")

        return points, colors

    def _load_ply(self, ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Simple PLY loader."""
        try:
            from plyfile import PlyData
            ply = PlyData.read(str(ply_path))
            vertex = ply['vertex']
            points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
            if 'red' in vertex:
                colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1)
            else:
                colors = np.ones_like(points) * 255
            return points, colors
        except ImportError:
            raise ImportError("plyfile library required for .ply files. Install with: pip install plyfile")

    def _load_camera_params(self, params_path: str) -> Dict:
        """Load camera parameters from JSON."""
        with open(params_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def project_mask_to_3d(
        self,
        mask_2d: np.ndarray,
        frame_idx: int,
        scaling_params: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Project 2D mask to 3D point cloud.

        Args:
            mask_2d: Binary 2D mask [H, W]
            frame_idx: Frame index
            scaling_params: Scaling parameters [scale_x, scale_y]

        Returns:
            Binary 3D mask [N_points] or None if projection fails
        """
        # Load depth map
        depth_map = self._load_depth_map(frame_idx)
        if depth_map is None:
            return None

        # Project 3D points to 2D
        projected_coords, point_depths = self._project_points_to_2d(frame_idx, scaling_params)

        if projected_coords is None:
            return None

        # Check which points are inside the 2D mask
        H, W = mask_2d.shape
        mask_3d = np.zeros(len(self.points), dtype=bool)

        for i, (u, v) in enumerate(projected_coords):
            # Check bounds
            if u < 0 or u >= W or v < 0 or v >= H:
                continue

            # Check if inside 2D mask
            if not mask_2d[int(v), int(u)]:
                continue

            # Check depth consistency (visibility)
            depth_at_pixel = depth_map[int(v), int(u)]
            if abs(point_depths[i] - depth_at_pixel) > self.vis_depth_threshold:
                continue

            mask_3d[i] = True

        return mask_3d

    def _load_depth_map(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load depth map for frame."""
        if frame_idx in self._depth_cache:
            return self._depth_cache[frame_idx]

        # Try common depth map naming conventions
        possible_names = [
            f"depth_{frame_idx:06d}.png",
            f"depth_{frame_idx:06d}.exr",
            f"depth_{frame_idx:04d}.png",
            f"{frame_idx:06d}_depth.png",
        ]

        for name in possible_names:
            depth_path = self.depth_maps_dir / name
            if depth_path.exists():
                depth_map = self._read_depth_file(depth_path)
                self._depth_cache[frame_idx] = depth_map
                return depth_map

        logger.warning(f"Depth map not found for frame {frame_idx}")
        return None

    def _read_depth_file(self, depth_path: Path) -> np.ndarray:
        """Read depth file (PNG or EXR)."""
        if depth_path.suffix == '.exr':
            try:
                import OpenEXR
                import Imath
                exr_file = OpenEXR.InputFile(str(depth_path))
                header = exr_file.header()
                dw = header['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                depth_str = exr_file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[::-1])
                return depth
            except ImportError:
                raise ImportError("OpenEXR library required for .exr files")
        else:
            # PNG depth (usually scaled)
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img).astype(np.float32) / 1000.0  # Assume mm to meters
            return depth

    def _project_points_to_2d(
        self,
        frame_idx: int,
        scaling_params: Optional[List[float]] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Project 3D points to 2D image coordinates.

        Args:
            frame_idx: Frame index
            scaling_params: Scaling parameters

        Returns:
            Tuple of (2D coordinates [N, 2], depths [N]) or None
        """
        if self.camera_params is None:
            logger.warning("Camera parameters not provided, using identity projection")
            return None

        # Get camera parameters for this frame
        if isinstance(self.camera_params, dict) and 'frames' in self.camera_params:
            frame_params = self.camera_params['frames'][frame_idx]
            K = np.array(frame_params['K'])  # Intrinsic matrix
            R = np.array(frame_params['R'])  # Rotation
            t = np.array(frame_params['t'])  # Translation
        else:
            # Use global parameters
            K = np.array(self.camera_params['K'])
            R = np.eye(3)
            t = np.zeros(3)

        # Transform to camera coordinates
        points_cam = (R @ self.points.T).T + t

        # Project to image
        points_2d_hom = (K @ points_cam.T).T
        depths = points_2d_hom[:, 2]

        # Normalize by depth
        coords = points_2d_hom[:, :2] / depths[:, None]

        # Apply scaling if provided
        if scaling_params is not None:
            coords[:, 0] *= scaling_params[0]
            coords[:, 1] *= scaling_params[1]

        return coords, depths


def main():
    """CLI test."""
    import argparse

    parser = argparse.ArgumentParser(description='Test mask projection')
    parser.add_argument('--pointcloud', required=True)
    parser.add_argument('--depth-maps-dir', required=True)
    parser.add_argument('--camera-params', required=True)
    parser.add_argument('--test-frame', type=int, default=0)

    args = parser.parse_args()

    projector = MaskProjector(
        pointcloud_path=args.pointcloud,
        depth_maps_dir=args.depth_maps_dir,
        camera_params_path=args.camera_params,
    )

    # Create dummy mask
    dummy_mask = np.zeros((480, 640), dtype=bool)
    dummy_mask[100:200, 200:400] = True

    mask_3d = projector.project_mask_to_3d(dummy_mask, args.test_frame)

    if mask_3d is not None:
        print(f"Projected {mask_3d.sum()} / {len(mask_3d)} points")
    else:
        print("Projection failed")


if __name__ == '__main__':
    main()
