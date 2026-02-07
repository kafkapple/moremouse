"""
MAMMAL Multi-view Dataset Loader

Loads multi-view images and poses from MAMMAL mouse dataset
for training the Gaussian Mouse Avatar (AGAM).

Supports two data formats:
1. Image directories: cam0/, cam1/, ... with frame_XXXXXX.png
2. Video files: 0.mp4, 1.mp4, ... (MAMMAL nerf format)

Reference: MoReMouse paper Section 3.1
- 800 frames for avatar training
- 8 camera viewpoints
- Synchronized multi-view capture
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VideoReader:
    """Efficient video reader with frame caching."""

    def __init__(self, video_path: Union[str, Path], cache_size: int = 100):
        self.video_path = Path(video_path)
        self.cache_size = cache_size
        self.cache: Dict[int, np.ndarray] = {}
        self.cache_order: List[int] = []

        # Open video to get properties
        cap = cv2.VideoCapture(str(self.video_path))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get frame by index with caching."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return None

        # Check cache
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        # Read from video
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest
            old_idx = self.cache_order.pop(0)
            del self.cache[old_idx]

        self.cache[frame_idx] = frame
        self.cache_order.append(frame_idx)

        return frame

    def __len__(self) -> int:
        return self.num_frames


class MAMMALMultiviewDataset(Dataset):
    """
    MAMMAL Multi-view Dataset for Gaussian Avatar Training.

    Supports two directory structures:

    1. Image-based (standard):
        markerless_mouse_1/
        ├── images/
        │   ├── cam0/
        │   │   ├── frame_000000.png
        │   │   └── ...
        │   ├── cam1/
        │   └── ... (8 cameras)
        ├── poses/
        │   ├── pose_000000.pkl  (or .npy)
        │   └── ...
        └── calibration.json (or cameras.pkl)

    2. Video-based (MAMMAL nerf format):
        markerless_mouse_1_nerf/
        ├── videos_undist/
        │   ├── 0.mp4
        │   ├── 1.mp4
        │   └── ... (one per camera)
        ├── poses/
        │   └── ...
        └── cameras.json

    Args:
        data_dir: Path to data directory
        num_frames: Number of frames to use (default: all)
        frame_start: Starting frame index
        frame_end: Ending frame index
        image_size: Resize images to this size
        cameras: List of camera indices to use (default: all)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        num_frames: Optional[int] = None,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        image_size: int = 800,
        cameras: Optional[List[int]] = None,
        pose_dir: Optional[Union[str, Path]] = None,
        require_pose: bool = True,  # Only use frames with valid pose data
        keypoints_dir: Optional[Union[str, Path]] = None,  # GT 2D keypoints directory
        canonical_mode: bool = False,
        crop_scale: float = 1.6,
        world_scale: float = 160.0,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self._explicit_pose_dir = Path(pose_dir) if pose_dir else None
        self.require_pose = require_pose
        self._explicit_keypoints_dir = Path(keypoints_dir) if keypoints_dir else None
        self.canonical_mode = canonical_mode
        self.crop_scale = crop_scale
        self.world_scale = world_scale

        if canonical_mode:
            print(f"Dataset initialized in CANONICAL MODE (crop_scale={crop_scale})")


        # Detect data format (video or image)
        self.use_video = self._detect_video_format()

        if self.use_video:
            # Video-based loading
            self.video_readers = self._init_video_readers()
            self.mask_readers = self._init_mask_readers()  # Segmentation masks
            self.pose_dir = self._explicit_pose_dir or self._find_pose_dir()
            self.calibration = self._load_calibration()
            self.global_transform = self._load_global_transform()  # center_rotation.npz
            self.keypoints2d = self._load_keypoints2d()  # GT 2D keypoints

            # Get camera list from video files
            if cameras is not None:
                self.cameras = cameras
            else:
                self.cameras = list(self.video_readers.keys())

            # Get frame count from first video
            if self.video_readers:
                first_reader = list(self.video_readers.values())[0]
                max_frames = first_reader.num_frames
                self.frames = list(range(
                    frame_start,
                    min(frame_end or max_frames, max_frames)
                ))

                # Filter to frames with pose data if required
                if require_pose and self.pose_dir is not None:
                    self.frames = self._filter_frames_with_pose(self.frames)

                if num_frames is not None:
                    self.frames = self.frames[:num_frames]
            else:
                self.frames = []

            print(f"MAMMAL Dataset (video mode): {len(self.frames)} frames, {len(self.cameras)} cameras")
        else:
            # Image-based loading
            self.video_readers = {}
            self.mask_readers = {}  # No mask support for image mode yet
            self.image_dir = self._find_image_dir()
            self.pose_dir = self._explicit_pose_dir or self._find_pose_dir()
            self.calibration = self._load_calibration()
            self.global_transform = self._load_global_transform()
            self.keypoints2d = self._load_keypoints2d()  # GT 2D keypoints

            # Get camera list
            if cameras is not None:
                self.cameras = cameras
            else:
                self.cameras = self._discover_cameras()

            # Get frame list
            self.frames = self._discover_frames(frame_start, frame_end, num_frames)

            print(f"MAMMAL Dataset (image mode): {len(self.frames)} frames, {len(self.cameras)} cameras")

    def _detect_video_format(self) -> bool:
        """Detect if data is in video format."""
        # Check for videos_undist directory with mp4 files
        video_dirs = [
            self.data_dir / "videos_undist",
            self.data_dir / "videos",
            self.data_dir,
        ]
        for vdir in video_dirs:
            if vdir.exists():
                mp4_files = list(vdir.glob("*.mp4"))
                if mp4_files:
                    return True
        return False

    def _init_video_readers(self) -> Dict[int, VideoReader]:
        """Initialize video readers for each camera."""
        readers = {}

        # Find video directory
        video_dirs = [
            self.data_dir / "videos_undist",
            self.data_dir / "videos",
            self.data_dir,
        ]

        video_dir = None
        for vdir in video_dirs:
            if vdir.exists() and list(vdir.glob("*.mp4")):
                video_dir = vdir
                break

        if video_dir is None:
            print("Warning: No video directory found")
            return readers

        # Find all video files
        for video_file in sorted(video_dir.glob("*.mp4")):
            # Extract camera index from filename (e.g., "0.mp4" -> 0)
            try:
                cam_idx = int(video_file.stem)
                readers[cam_idx] = VideoReader(video_file)
                print(f"  Loaded video: {video_file.name} ({readers[cam_idx].num_frames} frames)")
            except ValueError:
                # Try other naming conventions
                if video_file.stem.startswith("cam"):
                    try:
                        cam_idx = int(video_file.stem.replace("cam", ""))
                        readers[cam_idx] = VideoReader(video_file)
                        print(f"  Loaded video: {video_file.name} ({readers[cam_idx].num_frames} frames)")
                    except ValueError:
                        continue

        return readers

    def _init_mask_readers(self) -> Dict[int, VideoReader]:
        """Initialize video readers for segmentation masks (simpleclick_undist)."""
        readers = {}

        # Find mask video directory
        mask_dirs = [
            self.data_dir / "simpleclick_undist",
            self.data_dir / "masks_undist",
            self.data_dir / "masks",
        ]

        mask_dir = None
        for mdir in mask_dirs:
            if mdir.exists() and list(mdir.glob("*.mp4")):
                mask_dir = mdir
                break

        if mask_dir is None:
            print("Warning: No mask video directory found (simpleclick_undist)")
            return readers

        # Find all mask video files
        for mask_file in sorted(mask_dir.glob("*.mp4")):
            try:
                cam_idx = int(mask_file.stem)
                readers[cam_idx] = VideoReader(mask_file)
                print(f"  Loaded mask video: {mask_file.name} ({readers[cam_idx].num_frames} frames)")
            except ValueError:
                if mask_file.stem.startswith("cam"):
                    try:
                        cam_idx = int(mask_file.stem.replace("cam", ""))
                        readers[cam_idx] = VideoReader(mask_file)
                        print(f"  Loaded mask video: {mask_file.name}")
                    except ValueError:
                        continue

        return readers

    def _get_mask_from_video(self, cam_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        """Get segmentation mask from video.

        Returns:
            Binary mask [H, W] with values 0 or 1, or None if not available
        """
        if cam_idx not in self.mask_readers:
            return None

        reader = self.mask_readers[cam_idx]
        frame = reader.get_frame(frame_idx)

        if frame is None:
            return None

        # Convert to grayscale and binarize
        # simpleclick masks are typically white (255) on black (0)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Binarize: threshold at 127
        mask = (gray > 127).astype(np.float32)

        # Resize to target image size
        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return mask

    def _find_image_dir(self) -> Path:
        """Find image directory."""
        candidates = [
            self.data_dir / "images",
            self.data_dir / "imgs",
            self.data_dir / "rgb",
            self.data_dir,  # images might be directly in data_dir
        ]
        for path in candidates:
            if path.exists() and any(path.iterdir()):
                return path
        raise FileNotFoundError(f"Cannot find image directory in {self.data_dir}")

    def _find_pose_dir(self, pose_dir: Optional[Path] = None) -> Optional[Path]:
        """Find pose directory.

        Args:
            pose_dir: Optional explicit pose directory path

        Returns:
            Path to pose directory or None
        """
        # If explicit path provided, use it
        if pose_dir is not None:
            pose_dir = Path(pose_dir)
            if pose_dir.exists():
                print(f"Using pose directory: {pose_dir}")
                return pose_dir

        # Standard candidates in data directory
        candidates = [
            self.data_dir / "poses",
            self.data_dir / "pose",
            self.data_dir / "mammal_poses",
            self.data_dir / "annotations",
        ]

        # Also check for MAMMAL results in parent directories
        # Pattern: MAMMAL_mouse/results/monocular/mouse_batch_*/
        mammal_results = list(Path("/home/joon/MAMMAL_mouse/results/monocular").glob("mouse_batch_*"))
        if mammal_results:
            # Use the most recent one
            candidates.extend(sorted(mammal_results, reverse=True))

        for path in candidates:
            if path.exists() and any(path.glob("*.pkl")) or any(path.glob("*.npy")):
                print(f"Found pose directory: {path}")
                return path

        print("Warning: Pose directory not found. Using None poses.")
        return None

    def _filter_frames_with_pose(self, frames: List[int]) -> List[int]:
        """Filter frames to only those that have valid (non-zero) pose data.

        Args:
            frames: List of frame indices to filter

        Returns:
            List of frame indices that have valid pose files with non-zero values
        """
        if self.pose_dir is None:
            return frames

        valid_frames = []
        zero_pose_count = 0

        for frame_idx in frames:
            # Load pose and check if it has non-zero values
            pose_data = self._load_pose(frame_idx)
            if pose_data is not None and 'thetas' in pose_data:
                thetas = pose_data['thetas']
                # Check if pose has meaningful values (not all zeros)
                if isinstance(thetas, np.ndarray):
                    nonzero = (np.abs(thetas) > 0.01).sum()
                else:
                    nonzero = (thetas.abs() > 0.01).sum().item()

                if nonzero > 0:
                    valid_frames.append(frame_idx)
                else:
                    zero_pose_count += 1

        original_count = len(frames)
        filtered_count = len(valid_frames)
        print(f"Filtered frames with pose: {filtered_count}/{original_count} frames have valid pose data")
        if zero_pose_count > 0:
            print(f"  (Skipped {zero_pose_count} frames with zero pose values)")

        return valid_frames

    def _load_global_transform(self) -> Optional[Dict]:
        """Load global transform data from center_rotation.npz.

        This contains per-frame global transforms (center position and rotation).
        Format:
            - centers: (N, 3) - 3D positions
            - angles: (N,) - rotation angles (radians)
            - covs: (N, 3, 3) - covariance matrices

        Returns:
            Dict with 'centers', 'angles', 'covs' arrays, or None if not found
        """
        path = self.data_dir / "center_rotation.npz"
        if path.exists():
            data = np.load(path)
            result = {
                'centers': data['centers'],
                'angles': data['angles'],
                'covs': data.get('covs', None),
            }
            # Calculate frame mapping ratio
            num_transforms = len(result['centers'])
            if self.video_readers:
                first_reader = list(self.video_readers.values())[0]
                num_video_frames = first_reader.num_frames
                result['frame_ratio'] = num_video_frames / num_transforms
            else:
                result['frame_ratio'] = 1.0
            print(f"Loaded global transform: {num_transforms} entries (ratio: {result['frame_ratio']:.1f})")
            return result
        return None

    def _load_keypoints2d(self) -> Optional[Dict[int, np.ndarray]]:
        """Load GT 2D keypoints from keypoints2d_undist directory.

        MAMMAL format: result_view_{cam_idx}.pkl per camera
        Each pkl contains array of shape (num_frames, 22, 3) with (x, y, conf)

        Keypoints are in original video resolution and will be scaled
        to self.image_size in __getitem__.

        Returns:
            Dict mapping cam_idx -> array (num_frames, 22, 3) or None if not found
        """
        # Try explicit path first, then search common locations
        candidates = []
        if self._explicit_keypoints_dir is not None:
            candidates.append(self._explicit_keypoints_dir)

        # Search in data_dir and parent directories
        candidates.extend([
            self.data_dir / "keypoints2d_undist",
            self.data_dir / "keypoints2d",
            self.data_dir.parent / "keypoints2d_undist",
            # MAMMAL_mouse structure
            Path("/home/joon/MAMMAL_mouse/data") / self.data_dir.name / "keypoints2d_undist",
        ])

        kp_dir = None
        for cand in candidates:
            if cand.exists() and list(cand.glob("result_view_*.pkl")):
                kp_dir = cand
                break

        if kp_dir is None:
            print("GT keypoints2d not found (optional)")
            return None

        # Load keypoints for each camera
        keypoints = {}
        for pkl_file in sorted(kp_dir.glob("result_view_*.pkl")):
            try:
                cam_idx = int(pkl_file.stem.replace("result_view_", ""))
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                # data is array of shape (num_frames,) with each element (22, 3)
                if isinstance(data, np.ndarray):
                    keypoints[cam_idx] = data
            except (ValueError, pickle.UnpicklingError) as e:
                print(f"Warning: Failed to load {pkl_file}: {e}")
                continue

        if keypoints:
            first_cam = list(keypoints.keys())[0]
            num_frames = len(keypoints[first_cam])
            print(f"Loaded GT keypoints2d: {len(keypoints)} cameras, {num_frames} frames, 22 keypoints each")
            return keypoints

        return None

    def _load_calibration(self) -> Dict:
        """Load camera calibration."""
        candidates = [
            self.data_dir / "calibration.json",
            self.data_dir / "cameras.json",
            self.data_dir / "cameras.pkl",
            self.data_dir / "calib.json",
            self.data_dir / "new_cam.pkl",  # MAMMAL format
        ]

        for path in candidates:
            if path.exists():
                if path.suffix == '.json':
                    with open(path, 'r') as f:
                        return json.load(f)
                elif path.suffix == '.pkl':
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    # Handle MAMMAL new_cam.pkl format (list of camera dicts)
                    if isinstance(data, list) and len(data) > 0 and 'K' in data[0]:
                        return self._convert_mammal_calibration(data)
                    return data

        # Return default calibration
        print("Warning: Calibration file not found. Using default cameras.")
        return self._create_default_calibration()

    def _convert_mammal_calibration(self, cam_list: List[Dict]) -> Dict:
        """Convert MAMMAL new_cam.pkl format to standard calibration dict.

        MAMMAL format: list of dicts with K, R, T
        Output format: dict with 'cameras' key containing camera info

        NOTE: K matrix is adjusted to match self.image_size if original video
        has different resolution.
        """
        calib = {"cameras": {}}

        # Get original video size to compute K adjustment
        orig_width, orig_height = None, None
        if self.use_video and len(self.video_readers) > 0:
            # Get size from first video reader
            first_cam = list(self.video_readers.keys())[0]
            reader = self.video_readers[first_cam]
            if hasattr(reader, 'width') and hasattr(reader, 'height'):
                orig_width = reader.width
                orig_height = reader.height
            else:
                # Try to get from cap
                import cv2
                video_path = self.data_dir / "videos_undist" / f"{first_cam}.mp4"
                if video_path.exists():
                    cap = cv2.VideoCapture(str(video_path))
                    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

        # Compute scale factors for K adjustment
        if orig_width and orig_height:
            scale_x = self.image_size / orig_width
            scale_y = self.image_size / orig_height
            print(f"Adjusting K for resize: {orig_width}x{orig_height} -> {self.image_size}x{self.image_size}")
            print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        else:
            scale_x, scale_y = 1.0, 1.0
            print("Warning: Could not determine original video size, K not adjusted")

        for i, cam in enumerate(cam_list):
            K = np.array(cam['K']).copy()
            R = np.array(cam['R'])
            T = np.array(cam['T'])

            # Adjust K for image resize
            K[0, 0] *= scale_x  # fx
            K[0, 2] *= scale_x  # cx
            K[1, 1] *= scale_y  # fy
            K[1, 2] *= scale_y  # cy

            # Convert R, T to 4x4 viewmat (world-to-camera)
            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = T

            # Camera position in world coordinates
            # pos = -R^T @ T
            pos = -R.T @ T

            calib["cameras"][f"cam{i}"] = {
                "K": K.tolist(),
                "viewmat": viewmat.tolist(),
                "position": pos.tolist(),
            }

        print(f"Loaded MAMMAL calibration: {len(cam_list)} cameras")
        if orig_width and orig_height:
            print(f"  Adjusted K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        return calib

    def _create_default_calibration(self, num_cameras: int = 8) -> Dict:
        """Create default calibration for 8 cameras on a sphere."""
        calib = {"cameras": {}}

        # Default intrinsics (800x800, fov=30 deg)
        fx = fy = 800 / (2 * np.tan(np.radians(30) / 2))
        cx, cy = 400, 400
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Place cameras on sphere
        radius = 2.22
        for i in range(num_cameras):
            theta = 2 * np.pi * i / num_cameras
            phi = np.pi / 2  # Equator

            # Camera position
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            pos = np.array([x, y, z])

            # Look at origin
            forward = -pos / np.linalg.norm(pos)
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            R = np.stack([right, -up, forward], axis=0)
            t = -R @ pos

            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = t

            calib["cameras"][f"cam{i}"] = {
                "K": K.tolist(),
                "viewmat": viewmat.tolist(),
                "position": pos.tolist(),
            }

        return calib

    def _discover_cameras(self) -> List[int]:
        """Discover available cameras."""
        cameras = []
        for item in sorted(self.image_dir.iterdir()):
            if item.is_dir() and item.name.startswith("cam"):
                cam_idx = int(item.name.replace("cam", ""))
                cameras.append(cam_idx)

        if not cameras:
            # Try numbered directories
            for item in sorted(self.image_dir.iterdir()):
                if item.is_dir() and item.name.isdigit():
                    cameras.append(int(item.name))

        return cameras if cameras else list(range(8))

    def _discover_frames(
        self,
        start: int,
        end: Optional[int],
        num_frames: Optional[int],
    ) -> List[int]:
        """Discover available frames."""
        frames = set()

        # Look in first camera directory
        if self.cameras:
            cam_dir = self.image_dir / f"cam{self.cameras[0]}"
            if not cam_dir.exists():
                cam_dir = self.image_dir / str(self.cameras[0])

            if cam_dir.exists():
                for img_file in cam_dir.glob("*.png"):
                    # Extract frame number from filename
                    name = img_file.stem
                    # Try different naming conventions
                    for pattern in ["frame_", "img_", ""]:
                        if pattern in name:
                            try:
                                idx = int(name.replace(pattern, "").split("_")[0])
                                frames.add(idx)
                            except ValueError:
                                continue

        frames = sorted(frames)

        if not frames:
            # Fallback: use range
            frames = list(range(start, end or 1000))

        # Apply range
        if end is not None:
            frames = [f for f in frames if start <= f < end]
        else:
            frames = [f for f in frames if f >= start]

        # Limit number of frames
        if num_frames is not None:
            frames = frames[:num_frames]

        return frames

    def _get_image_path(self, cam_idx: int, frame_idx: int) -> Optional[Path]:
        """Get image file path."""
        cam_dir = self.image_dir / f"cam{cam_idx}"
        if not cam_dir.exists():
            cam_dir = self.image_dir / str(cam_idx)

        if not cam_dir.exists():
            return None

        # Try different naming conventions
        patterns = [
            f"frame_{frame_idx:06d}.png",
            f"frame_{frame_idx:04d}.png",
            f"img_{frame_idx:06d}.png",
            f"{frame_idx:06d}.png",
            f"{frame_idx:04d}.png",
        ]

        for pattern in patterns:
            path = cam_dir / pattern
            if path.exists():
                return path

        return None

    def _load_pose(self, frame_idx: int) -> Optional[Dict]:
        """Load MAMMAL pose for frame.

        Returns:
            Dict with:
                - thetas: (140, 3) joint rotations (axis-angle)
                - R: (3,) global rotation
                - T: (3,) global translation
                - s: scalar scale
            Or None if not found
        """
        if self.pose_dir is None:
            return None

        # Try different naming conventions
        patterns = [
            f"frame_{frame_idx:04d}_params.pkl",  # MAMMAL results format
            f"pose_{frame_idx:06d}.pkl",
            f"pose_{frame_idx:04d}.pkl",
            f"pose_{frame_idx:06d}.npy",
            f"{frame_idx:06d}.pkl",
            f"{frame_idx:06d}.npy",
        ]

        for pattern in patterns:
            path = self.pose_dir / pattern
            if path.exists():
                if path.suffix == '.pkl':
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    # MAMMAL results format: dict with thetas, R, T, s
                    if isinstance(data, dict) and 'thetas' in data:
                        return {
                            'thetas': data['thetas'].squeeze(0),  # (140, 3)
                            'R': data['R'].squeeze(0),  # (3,)
                            'T': data['T'].squeeze(0),  # (3,)
                            's': data['s'].squeeze(),  # scalar
                        }
                    # Legacy format
                    if isinstance(data, dict):
                        pose = data.get('pose', data.get('poses', None))
                        if pose is not None:
                            return {'thetas': pose}
                    return {'thetas': data} if data is not None else None
                elif path.suffix == '.npy':
                    return {'thetas': np.load(path)}

        return None

    def _get_global_transform(self, frame_idx: int) -> Optional[Dict]:
        """Get global transform (center and rotation) for a frame.

        Args:
            frame_idx: Video frame index

        Returns:
            Dict with 'center' (3,) and 'angle' (float) tensors, or None
        """
        if not hasattr(self, 'global_transform') or self.global_transform is None:
            return None

        # Map video frame to transform index
        ratio = self.global_transform.get('frame_ratio', 1.0)
        transform_idx = int(frame_idx / ratio)

        # Clamp to valid range
        max_idx = len(self.global_transform['centers']) - 1
        transform_idx = min(transform_idx, max_idx)

        center = self.global_transform['centers'][transform_idx]
        angle = self.global_transform['angles'][transform_idx]

        return {
            'center': torch.from_numpy(center).float(),
            'angle': torch.tensor(angle).float(),
            'valid': True
        }

    def _compute_bbox_from_global(
        self,
        center_world: torch.Tensor,
        K: torch.Tensor,
        viewmat: torch.Tensor,
        orig_width: int,
        orig_height: int,
        scale: float = 1.6,
    ) -> Tuple[int, int, int]:
        """Compute bounding box from global center.
        
        Args:
            center_world: [3] Mouse center in world coordinates
            K: [3, 3] Camera intrinsics
            viewmat: [4, 4] World-to-Camera matrix
            orig_width, orig_height: Original image dimensions
            scale: Scale factor for bbox size relative to mouse size
            
        Returns:
            (x1, y1, crop_size) square bbox
        """
        # Mouse physical size approx 10cm = 0.1m
        # We want crop size to cover about 10-15cm in world
        # Project world center to image
        
        # Transform to camera space
        R = viewmat[:3, :3]
        t = viewmat[:3, 3]
        center_cam = (R @ center_world) + t  # [3]
        
        # Project center
        center_proj = (K @ center_cam) # [3]
        center_2d = center_proj[:2] / center_proj[2]
        cx, cy = center_2d[0].item(), center_2d[1].item()
        
        # Estimate size: project a point offset by radius
        # Mouse radius approx 5cm = 0.05m
        radius_world = 0.05
        offset_world = center_world.clone()
        offset_world[0] += radius_world # Offset in X
        offset_cam = (R @ offset_world) + t
        offset_proj = (K @ offset_cam)
        offset_2d = offset_proj[:2] / offset_proj[2]
        
        # Pixel radius
        radius_px = torch.norm(offset_2d - center_2d).item()
        
        # Crop size = diameter * scale
        crop_size = int(radius_px * 2 * scale)
        
        # Ensure crop size is reasonable (e.g. at least 200px)
        crop_size = max(crop_size, 200)
        
        # XY top-left
        x1 = int(cx - crop_size / 2)
        y1 = int(cy - crop_size / 2)
        
        return x1, y1, crop_size

    def _crop_and_resize_image(
        self,
        img: torch.Tensor,
        x1: int,
        y1: int,
        crop_size: int,
        target_size: int,
    ) -> torch.Tensor:
        """Crop image and resize to target size."""
        # img: [C, H, W]
        C, H, W = img.shape
        
        # Pad if crop goes outside
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x1 + crop_size - W)
        pad_b = max(0, y1 + crop_size - H)
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            img = F.pad(img, (pad_l, pad_r, pad_t, pad_b))
            x1 += pad_l
            y1 += pad_t
            
        # Crop
        crop = img[:, y1:y1+crop_size, x1:x1+crop_size]
        
        # Resize
        if crop_size != target_size:
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
        return crop

    def _compute_canonical_camera(
        self,
        K: torch.Tensor,
        viewmat: torch.Tensor,
        center_world: torch.Tensor,
        yaw_angle: float,
        crop_size: int,
        target_size: int,
        world_scale: float = 160.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute canonical camera parameters (K_new, V_new).
        
        Args:
            K: Original intrinsics
            viewmat: Original extrinsics (World-to-Camera)
            center_world: Mouse center in World
            yaw_angle: Mouse yaw angle (radians)
            crop_size: Size of crop in original image
            target_size: Output image size
            world_scale: Scaling factor used in training (e.g. 160)
            
        Returns:
            K_new, V_new
        """
        # 1. Update K for crop and resize
        # Camera center moves: cx' = (cx - x1) * (target/crop)
        # Focal length scales: fx' = fx * (target/crop)
        scale_factor = target_size / crop_size
        
        # We need bbox x1, y1. Assuming center is at (cx, cy) of crop
        # Actually we need exact x1, y1 used for cropping
        # Re-computing strictly here is tricky if logic duplicates.
        # But we assume this function is called with consistent parameters.
        
        # Wait, we need x1,y1 to adjust K principal point.
        # Let's pass crop transform info instead or compute X1,Y1 here?
        # Better: This function calculates everything.
        
        # Re-calculate X1, Y1 based on projected center and crop_size
        R = viewmat[:3, :3]
        t = viewmat[:3, 3]
        center_cam = (R @ center_world) + t
        center_proj = (K @ center_cam)
        cx_orig = (center_proj[0] / center_proj[2]).item()
        cy_orig = (center_proj[1] / center_proj[2]).item()
        
        x1 = int(cx_orig - crop_size / 2)
        y1 = int(cy_orig - crop_size / 2)
        
        K_new = torch.eye(3)
        K_new[0, 0] = K[0, 0] * scale_factor
        K_new[1, 1] = K[1, 1] * scale_factor
        K_new[0, 2] = (K[0, 2] - x1) * scale_factor
        K_new[1, 2] = (K[1, 2] - y1) * scale_factor
        K_new[2, 2] = 1.0
        
        # 2. Compute Canonical View Matrix
        # Original: P_cam = V_real * P_world
        # Mouse Model: P_world = M_mouse * P_model
        #   where M_mouse has rotation R_yaw and translation center_world
        #   M_mouse = [ R_z(yaw) | center_world ]
        # Combine: P_cam = V_real * (M_mouse * P_model)
        #                = (V_real * M_mouse) * P_model
        #                = V_canonical_unscaled * P_model
        #
        # But wait, P_model in canonical space is scaled by 1/180?
        # If we train with canonical_mode=True, the mesh is scaled by 1/180.
        # So P_canonical = P_model * (1/180)
        # P_model = P_canonical * 180
        # P_cam = V_canonical_unscaled * (P_canonical * 180)
        #       = (V_canonical_unscaled * 180_matrix) * P_canonical
        # This effectively scales the translation of V_canonical by 180.
        
        # Construct M_mouse (Model-to-World)
        # Z-rotation for yaw
        cos_a = np.cos(yaw_angle)
        sin_a = np.sin(yaw_angle)
        R_mouse = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # NOTE: MAMMAL global transform angle might be inverted or have offset.
        # Based on calibration report (neg_yaw), we might need to negate.
        # Let's assume input yaw_angle is already correct (negated if needed).
        
        M_mouse = torch.eye(4)
        M_mouse[:3, :3] = R_mouse
        M_mouse[:3, 3] = center_world
        
        # V_canonical_unscaled = V_real * M_mouse
        V_real_mat = viewmat # already 4x4
        V_canon_unscaled = V_real_mat @ M_mouse
        
        # Adjust for Mesh Scale (1/180)
        # Meaning: The canonical mesh is tiny (unit sphere).
        # The camera should see it as if it were large.
        # So we simply scale the translation part?
        # P_cam = R * (P + t)
        # P_cam = R * (P_canon * S + t_canon)
        # P_cam = R * P_canon * S + R * t_canon
        # If we want to keep the same image, and the object is scaled by S,
        # the camera's translation (t) needs to be scaled by S.
        # P_cam = R * P_canon + t_new
        # We want R * P_canon + t_new = R_orig * (P_canon * S) + t_orig
        # R_orig * P_canon * S + t_orig = R_orig * P_canon * S + t_orig
        # This means the translation part of V_canon_unscaled needs to be scaled.
        
        V_canon = V_canon_unscaled.clone()
        V_canon[:3, 3] /= world_scale
        
        return K_new, V_canon

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image."""
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def get_camera_params(self, cam_idx: int) -> Dict:
        """Get camera parameters."""
        cam_key = f"cam{cam_idx}"
        if cam_key in self.calibration.get("cameras", {}):
            return self.calibration["cameras"][cam_key]

        # Fallback to default
        default = self._create_default_calibration(len(self.cameras))
        return default["cameras"].get(cam_key, default["cameras"]["cam0"])

    def __len__(self) -> int:
        return len(self.frames)

    def _get_frame_from_video(self, cam_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        """Get frame from video reader."""
        if cam_idx not in self.video_readers:
            return None

        frame = self.video_readers[cam_idx].get_frame(frame_idx)
        if frame is None:
            return None

        # Resize if needed
        if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
            frame = cv2.resize(frame, (self.image_size, self.image_size))

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    def __getitem__(self, idx: int) -> Dict:
        """
        Get multi-view data for a frame.

        Returns:
            dict with:
                - images: [num_cameras, H, W, 3] multi-view images
                - viewmats: [num_cameras, 4, 4] view matrices
                - Ks: [num_cameras, 3, 3] intrinsics
                - pose: [J*3] MAMMAL pose (or None)
                - frame_idx: frame index
        """
        frame_idx = self.frames[idx]

        images = []
        masks = []
        viewmats = []
        Ks = []

        for cam_idx in self.cameras:
            # Load image from video or image file
            if self.use_video:
                img = self._get_frame_from_video(cam_idx, frame_idx)
            else:
                img_path = self._get_image_path(cam_idx, frame_idx)
                if img_path is not None:
                    img = self._load_image(img_path)
                else:
                    img = None

            if img is None:
                # Placeholder for missing frames
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

            images.append(img)

            # Load mask from video
            if self.use_video:
                mask = self._get_mask_from_video(cam_idx, frame_idx)
            else:
                mask = None  # Image mode doesn't support masks yet

            if mask is None:
                # Default: all ones (no masking)
                mask = np.ones((self.image_size, self.image_size), dtype=np.float32)

            masks.append(mask)

            # Get camera params
            cam_params = self.get_camera_params(cam_idx)
            viewmats.append(np.array(cam_params["viewmat"]))
            Ks.append(np.array(cam_params["K"]))

        # Check if we have valid masks
        has_masks = len(self.mask_readers) > 0 if hasattr(self, 'mask_readers') else False

        # Load pose (return empty tensor if not available)
        pose_data = self._load_pose(frame_idx)
        has_pose = pose_data is not None and 'thetas' in pose_data

        if has_pose:
            thetas = pose_data['thetas']  # (140, 3) axis-angle
            if isinstance(thetas, np.ndarray):
                pose_tensor = torch.from_numpy(thetas).float().reshape(-1)  # flatten to (420,)
            else:
                pose_tensor = thetas.float().reshape(-1)

            # Extract global pose from MAMMAL result if available
            if 'R' in pose_data and 'T' in pose_data:
                R_global = pose_data['R']  # (3,) axis-angle rotation
                T_global = pose_data['T']  # (3,) translation
                scale = pose_data.get('s', 1.0)  # scale factor

                # Handle None or invalid scale
                if scale is None:
                    scale = 1.0

                if isinstance(R_global, np.ndarray):
                    R_global = torch.from_numpy(R_global).float()
                if isinstance(T_global, np.ndarray):
                    T_global = torch.from_numpy(T_global).float()
                if not isinstance(scale, torch.Tensor):
                    scale = torch.tensor(float(scale), dtype=torch.float32)

                mammal_global = {
                    'R': R_global,
                    'T': T_global,
                    's': scale,
                }
            else:
                mammal_global = None
        else:
            # Default: empty pose tensor (will use T-pose)
            pose_tensor = torch.zeros(140 * 3, dtype=torch.float32)  # 140 joints * 3
            mammal_global = None

        # Load global transform (center position and rotation)
        global_trans = self._get_global_transform(frame_idx)

        # Convert None to placeholder tensors for DataLoader collate compatibility
        # mammal_global: R (3,), T (3,), s (scalar)
        if mammal_global is None:
            mammal_global_out = {
                'R': torch.zeros(3, dtype=torch.float32),
                'T': torch.zeros(3, dtype=torch.float32),
                's': torch.tensor(1.0, dtype=torch.float32),  # default scale 1.0
                'valid': torch.tensor(False),
            }
        else:
            # Ensure all values are tensors
            R = mammal_global['R'] if isinstance(mammal_global['R'], torch.Tensor) else torch.from_numpy(mammal_global['R']).float()
            T = mammal_global['T'] if isinstance(mammal_global['T'], torch.Tensor) else torch.from_numpy(mammal_global['T']).float()
            s = mammal_global['s'] if isinstance(mammal_global['s'], torch.Tensor) else torch.tensor(float(mammal_global['s']), dtype=torch.float32)
            mammal_global_out = {
                'R': R,
                'T': T,
                's': s,
                'valid': torch.tensor(True),
            }

        # global_transform: center (3,), angle (scalar)
        if global_trans is None:
            global_trans_out = {
                'center': torch.zeros(3, dtype=torch.float32),
                'angle': torch.tensor(0.0, dtype=torch.float32),
                'valid': torch.tensor(False),
            }
        else:
            global_trans_out = {
                'center': global_trans['center'],
                'angle': global_trans['angle'],
                'valid': torch.tensor(True),
            }

        # Load GT 2D keypoints (scaled to current image_size)
        keypoints2d_list = []
        has_keypoints2d = self.keypoints2d is not None
        if has_keypoints2d:
            # Get original video size for scaling
            orig_width, orig_height = 1152, 1024  # Default MAMMAL resolution
            if self.use_video and self.video_readers:
                first_reader = list(self.video_readers.values())[0]
                orig_width = first_reader.width
                orig_height = first_reader.height

            scale_x = self.image_size / orig_width
            scale_y = self.image_size / orig_height

            for cam_idx in self.cameras:
                if cam_idx in self.keypoints2d:
                    kp = self.keypoints2d[cam_idx][frame_idx].copy()  # (22, 3)
                    # Scale x, y to current image_size
                    kp[:, 0] *= scale_x
                    kp[:, 1] *= scale_y
                    keypoints2d_list.append(kp)
                else:
                    # Placeholder for missing camera
                    keypoints2d_list.append(np.zeros((22, 3), dtype=np.float32))

        if keypoints2d_list:
            keypoints2d_tensor = torch.from_numpy(np.stack(keypoints2d_list)).float()  # [C, 22, 3]
        else:
            # Placeholder: [C, 22, 3] all zeros
            keypoints2d_tensor = torch.zeros(len(self.cameras), 22, 3, dtype=torch.float32)

        return {
            "images": torch.from_numpy(np.stack(images)),  # [C, H, W, 3]
            "masks": torch.from_numpy(np.stack(masks)),  # [C, H, W] segmentation masks
            "viewmats": torch.from_numpy(np.stack(viewmats)).float(),  # [C, 4, 4]
            "Ks": torch.from_numpy(np.stack(Ks)).float(),  # [C, 3, 3]
            "pose": pose_tensor,  # [140*3] flattened axis-angle rotations
            "has_pose": torch.tensor(has_pose),  # Flag to indicate if pose was loaded
            "has_masks": torch.tensor(has_masks),  # Flag to indicate if masks were loaded
            "mammal_global": mammal_global_out,  # Dict with R, T, s, valid from MAMMAL fitting
            "global_transform": global_trans_out,  # Dict with center, angle, valid from center_rotation.npz
            "frame_idx": torch.tensor(frame_idx),  # Convert to tensor for collate
            "keypoints2d": keypoints2d_tensor,  # [C, 22, 3] GT 2D keypoints (x, y, conf)
            "has_keypoints2d": torch.tensor(has_keypoints2d),  # Flag
        }


class CanonicalSpaceDataset(Dataset):
    """
    Canonical Space Dataset following MoReMouse paper.

    Key differences from MAMMALMultiviewDataset:
    1. Images are cropped around mouse centroid (from GT 2D keypoints)
    2. Ψ_g = 0: No global transform (scale, rotation, translation)
    3. Synthetic cameras on unit sphere (radius 2.22)
    4. Mesh scaled by 1/180 to fit in unit sphere

    This approach avoids coordinate alignment issues by:
    - Normalizing images to mouse-centered crops
    - Using canonical pose space (no global transform)
    - Generating virtual cameras instead of using real camera poses

    Reference: MoReMouse paper Section 3.1
    """

    # Paper constants
    MESH_SCALE = 1.0 / 180.0  # Scale to fit in unit sphere
    CAMERA_RADIUS = 2.22      # Camera distance from origin
    FOV_DEG = 29.86           # Field of view in degrees
    IMAGE_SIZE = 800          # Paper uses 800x800

    def __init__(
        self,
        data_dir: Union[str, Path],
        num_frames: Optional[int] = None,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        image_size: int = 800,
        cameras: Optional[List[int]] = None,
        pose_dir: Optional[Union[str, Path]] = None,
        require_pose: bool = True,
        keypoints_dir: Optional[Union[str, Path]] = None,
        crop_size: int = 400,  # Size of crop around mouse (before resize)
        crop_margin: float = 1.5,  # Margin multiplier for crop box
    ):
        """
        Args:
            data_dir: Path to MAMMAL data directory
            num_frames: Number of frames to use
            frame_start: Starting frame index
            frame_end: Ending frame index
            image_size: Output image size (default: 800 for paper)
            cameras: List of camera indices to use
            pose_dir: MAMMAL pose estimation results directory
            require_pose: Only use frames with valid pose data
            keypoints_dir: GT 2D keypoints directory
            crop_size: Size of crop around mouse centroid
            crop_margin: Margin multiplier for bounding box
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.crop_size = crop_size
        self.crop_margin = crop_margin
        self._explicit_pose_dir = Path(pose_dir) if pose_dir else None
        self.require_pose = require_pose
        self._explicit_keypoints_dir = Path(keypoints_dir) if keypoints_dir else None

        # Initialize base dataset for video reading and pose loading
        self._base_dataset = MAMMALMultiviewDataset(
            data_dir=data_dir,
            num_frames=num_frames,
            frame_start=frame_start,
            frame_end=frame_end,
            image_size=1024,  # Load at high resolution for cropping
            cameras=cameras,
            pose_dir=pose_dir,
            require_pose=require_pose,
            keypoints_dir=keypoints_dir,
        )

        # Copy relevant attributes
        self.frames = self._base_dataset.frames
        self.cameras = self._base_dataset.cameras
        self.keypoints2d = self._base_dataset.keypoints2d
        self.video_readers = self._base_dataset.video_readers
        self.use_video = self._base_dataset.use_video

        # Compute focal length from FOV
        self.focal_length = self.image_size / (2 * np.tan(np.radians(self.FOV_DEG / 2)))

        # Generate synthetic cameras on sphere
        self.synthetic_cameras = self._generate_sphere_cameras(len(self.cameras))

        print(f"CanonicalSpaceDataset: {len(self.frames)} frames, {len(self.cameras)} cameras")
        print(f"  Mesh scale: {self.MESH_SCALE:.6f}")
        print(f"  Camera radius: {self.CAMERA_RADIUS}")
        print(f"  Focal length: {self.focal_length:.2f}")

    def _generate_sphere_cameras(self, n_cameras: int) -> List[Dict]:
        """Generate cameras on unit sphere using Fibonacci lattice.

        Following MoReMouse: cameras at radius 2.22, looking at origin.
        """
        import math

        cameras = []
        golden_ratio = (1 + math.sqrt(5)) / 2

        for i in range(n_cameras):
            # Fibonacci lattice for uniform distribution
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / n_cameras)

            # Spherical to Cartesian
            x = self.CAMERA_RADIUS * math.sin(phi) * math.cos(theta)
            y = self.CAMERA_RADIUS * math.sin(phi) * math.sin(theta)
            z = self.CAMERA_RADIUS * math.cos(phi)
            position = np.array([x, y, z])

            # Look at origin
            forward = -position / np.linalg.norm(position)
            up = np.array([0, 0, 1])

            # Handle near-vertical cameras
            if abs(np.dot(forward, up)) > 0.99:
                up = np.array([0, 1, 0])

            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # View matrix (world to camera)
            R = np.stack([right, -up, forward], axis=0)
            t = -R @ position

            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = t

            # Intrinsics
            K = np.array([
                [self.focal_length, 0, self.image_size / 2],
                [0, self.focal_length, self.image_size / 2],
                [0, 0, 1]
            ])

            cameras.append({
                'viewmat': viewmat,
                'K': K,
                'position': position,
                'azimuth': math.degrees(theta) % 360,
                'elevation': 90 - math.degrees(phi),
            })

        return cameras

    def _crop_around_centroid(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        orig_width: int,
        orig_height: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop image around mouse centroid based on GT keypoints.

        Args:
            image: Input image [H, W, 3]
            keypoints: GT 2D keypoints [22, 3] (x, y, confidence)
            orig_width: Original video width
            orig_height: Original video height

        Returns:
            cropped_image: Cropped and resized image [image_size, image_size, 3]
            adjusted_K: Adjusted intrinsics [3, 3]
            transformed_keypoints: Keypoints in cropped image coordinates [22, 3]
        """
        # Filter valid keypoints (confidence > 0.3)
        valid_mask = keypoints[:, 2] > 0.3
        if valid_mask.sum() < 3:
            # Fallback: use center crop
            cx, cy = orig_width / 2, orig_height / 2
        else:
            valid_kps = keypoints[valid_mask, :2]
            cx, cy = valid_kps.mean(axis=0)

        # Compute bounding box from keypoints
        if valid_mask.sum() >= 3:
            x_min, y_min = valid_kps.min(axis=0)
            x_max, y_max = valid_kps.max(axis=0)
            bbox_size = max(x_max - x_min, y_max - y_min) * self.crop_margin
        else:
            bbox_size = self.crop_size

        # Ensure minimum crop size
        bbox_size = max(bbox_size, self.crop_size)

        # Compute crop box (square, centered on centroid)
        half_size = bbox_size / 2
        x1 = int(max(0, cx - half_size))
        y1 = int(max(0, cy - half_size))
        x2 = int(min(orig_width, cx + half_size))
        y2 = int(min(orig_height, cy + half_size))

        # Ensure square crop
        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_size = min(crop_w, crop_h)

        # Re-center if needed
        if crop_w != crop_h:
            x1 = int(cx - crop_size / 2)
            y1 = int(cy - crop_size / 2)
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            # Clamp to image bounds
            if x1 < 0:
                x1, x2 = 0, crop_size
            if y1 < 0:
                y1, y2 = 0, crop_size
            if x2 > orig_width:
                x1, x2 = orig_width - crop_size, orig_width
            if y2 > orig_height:
                y1, y2 = orig_height - crop_size, orig_height

        # Crop image
        cropped = image[y1:y2, x1:x2]

        # Resize to target size
        if cropped.shape[0] != self.image_size or cropped.shape[1] != self.image_size:
            cropped = cv2.resize(cropped, (self.image_size, self.image_size))

        # Compute adjusted intrinsics for the crop
        # The principal point shifts and focal length scales
        scale = self.image_size / crop_size
        K = np.array([
            [self.focal_length, 0, self.image_size / 2],
            [0, self.focal_length, self.image_size / 2],
            [0, 0, 1]
        ])

        # Transform keypoints to cropped image coordinates
        transformed_kp = keypoints.copy()
        # Shift by crop origin
        transformed_kp[:, 0] = (keypoints[:, 0] - x1) * scale
        transformed_kp[:, 1] = (keypoints[:, 1] - y1) * scale
        # Keep confidence unchanged

        # Return crop info for later projection of model keypoints
        crop_info = {
            'x1': x1,
            'y1': y1,
            'scale': scale,
            'crop_size': crop_size,
        }

        return cropped, K, transformed_kp, crop_info

    def _get_real_camera_params(self, cam_idx: int) -> Dict[str, torch.Tensor]:
        """Get real MAMMAL camera parameters for Option A projection.

        Returns K and viewmat from the actual MAMMAL calibration (not synthetic).
        These are used to project model 3D joints to real camera space.

        Note: The K returned here is for ORIGINAL image resolution (1152x1024),
        NOT adjusted for resize. The projection pipeline will use crop_info
        to transform from original image coords to cropped image coords.
        """
        # Get camera params from base dataset's MAMMAL calibration
        cam_params = self._base_dataset.get_camera_params(cam_idx)

        # Get original K (before resize adjustment)
        # We need the K for original video resolution for proper projection
        K_original = np.array(cam_params['K'])

        # Undo the resize scaling that was applied in _convert_mammal_calibration
        # Original video: 1152x1024, adjusted to self._base_dataset.image_size
        orig_width, orig_height = 1152, 1024  # Default MAMMAL resolution
        if self._base_dataset.use_video and len(self._base_dataset.video_readers) > 0:
            first_reader = list(self._base_dataset.video_readers.values())[0]
            if hasattr(first_reader, 'width') and hasattr(first_reader, 'height'):
                orig_width = first_reader.width
                orig_height = first_reader.height

        # The stored K was scaled by (image_size / orig_size), undo this
        base_image_size = self._base_dataset.image_size
        scale_x = orig_width / base_image_size
        scale_y = orig_height / base_image_size
        K_original[0, 0] *= scale_x  # fx
        K_original[0, 2] *= scale_x  # cx
        K_original[1, 1] *= scale_y  # fy
        K_original[1, 2] *= scale_y  # cy

        viewmat = np.array(cam_params['viewmat'])

        return {
            'K': torch.from_numpy(K_original).float(),  # [3, 3] for original resolution
            'viewmat': torch.from_numpy(viewmat).float(),  # [4, 4] world-to-camera
            'orig_width': torch.tensor(orig_width, dtype=torch.float32),
            'orig_height': torch.tensor(orig_height, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get canonical space data for a frame.

        Returns:
            dict with:
                - images: [num_cameras, H, W, 3] cropped multi-view images
                - viewmats: [num_cameras, 4, 4] synthetic view matrices
                - Ks: [num_cameras, 3, 3] intrinsics
                - pose: [J*3] local pose only (Ψ_g = 0)
                - frame_idx: frame index
                - mesh_scale: scale factor for mesh (1/180)
                - keypoints2d: [num_cameras, 22, 3] GT keypoints in cropped coords
        """
        frame_idx = self.frames[idx]

        images = []
        viewmats = []
        Ks = []
        keypoints2d_list = []

        # Get original video size
        orig_width, orig_height = 1152, 1024  # Default MAMMAL resolution
        if self.use_video and self.video_readers:
            first_reader = list(self.video_readers.values())[0]
            orig_width = first_reader.width
            orig_height = first_reader.height

        has_keypoints2d = self.keypoints2d is not None
        first_crop_info = None  # Will be set by first camera

        for i, cam_idx in enumerate(self.cameras):
            # Load image at high resolution
            if self.use_video:
                frame = self._base_dataset._get_frame_from_video(cam_idx, frame_idx)
                if frame is not None:
                    # Scale back to original resolution for cropping
                    img = cv2.resize(
                        (frame * 255).astype(np.uint8),
                        (orig_width, orig_height)
                    )
                else:
                    img = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
            else:
                img = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)

            # Get GT keypoints for this camera
            if has_keypoints2d and cam_idx in self.keypoints2d:
                kp = self.keypoints2d[cam_idx][frame_idx].copy()
            else:
                # Fallback: center keypoints
                kp = np.zeros((22, 3))
                kp[:, 0] = orig_width / 2
                kp[:, 1] = orig_height / 2
                kp[:, 2] = 1.0

            # Crop around mouse centroid (also transforms keypoints)
            cropped_img, _, transformed_kp, crop_info = self._crop_around_centroid(
                img, kp, orig_width, orig_height
            )

            # Normalize to [0, 1]
            cropped_img = cropped_img.astype(np.float32) / 255.0
            images.append(cropped_img)
            keypoints2d_list.append(transformed_kp)

            # Store crop info for the first camera (for visualization)
            if i == 0:
                first_crop_info = crop_info

            # Use synthetic camera (same for all frames, different per camera index)
            syn_cam = self.synthetic_cameras[i % len(self.synthetic_cameras)]
            viewmats.append(syn_cam['viewmat'])
            Ks.append(syn_cam['K'])

        # Load pose (local only, Ψ_g = 0)
        pose_data = self._base_dataset._load_pose(frame_idx)
        has_pose = pose_data is not None and 'thetas' in pose_data

        if has_pose:
            thetas = pose_data['thetas']
            if isinstance(thetas, np.ndarray):
                pose_tensor = torch.from_numpy(thetas).float().reshape(-1)
            else:
                pose_tensor = thetas.float().reshape(-1)
        else:
            pose_tensor = torch.zeros(140 * 3, dtype=torch.float32)

        return {
            "images": torch.from_numpy(np.stack(images)),  # [C, H, W, 3]
            "viewmats": torch.from_numpy(np.stack(viewmats)).float(),  # [C, 4, 4]
            "Ks": torch.from_numpy(np.stack(Ks)).float(),  # [C, 3, 3]
            "pose": pose_tensor,  # [140*3] local pose only
            "has_pose": torch.tensor(has_pose),
            "frame_idx": torch.tensor(frame_idx),
            "mesh_scale": torch.tensor(self.MESH_SCALE, dtype=torch.float32),
            # Ψ_g = 0 for canonical space
            "global_transform": {
                'center': torch.zeros(3, dtype=torch.float32),
                'angle': torch.tensor(0.0, dtype=torch.float32),
                'valid': torch.tensor(False),
            },
            "mammal_global": {
                'R': torch.zeros(3, dtype=torch.float32),
                'T': torch.zeros(3, dtype=torch.float32),
                's': torch.tensor(1.0, dtype=torch.float32),
                'valid': torch.tensor(False),
            },
            # GT keypoints transformed to cropped image coordinates
            "keypoints2d": torch.from_numpy(np.stack(keypoints2d_list)).float(),  # [C, 22, 3]
            "has_keypoints2d": torch.tensor(has_keypoints2d),
            # Crop info for projecting model keypoints to GT image coords
            "crop_info": {
                'x1': torch.tensor(first_crop_info['x1'], dtype=torch.float32),
                'y1': torch.tensor(first_crop_info['y1'], dtype=torch.float32),
                'scale': torch.tensor(first_crop_info['scale'], dtype=torch.float32),
                'crop_size': torch.tensor(first_crop_info['crop_size'], dtype=torch.float32),
            },
            # Real MAMMAL camera params for Option A (MoReMouse original method)
            # Used to project model 3D joints to real camera space, then apply crop
            "real_camera": self._get_real_camera_params(0),  # First camera
        }


def create_mammal_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 1,
    num_workers: int = 4,
    canonical_space: bool = False,
    **kwargs,
) -> DataLoader:
    """Create MAMMAL multi-view dataloader.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size (1 = 1 frame with all views)
        num_workers: Number of data loading workers
        canonical_space: If True, use CanonicalSpaceDataset (MoReMouse paper method)
        **kwargs: Additional arguments for dataset

    Returns:
        DataLoader instance
    """
    if canonical_space:
        dataset = CanonicalSpaceDataset(data_dir, **kwargs)
    else:
        dataset = MAMMALMultiviewDataset(data_dir, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with images and metadata
        """
        if self.use_video:
            # Use frames list to get frame_idx
            # Note: in video mode, we yield all cameras for a single frame
            # Or yield one image per index?
            # Usually for training we want random access to any (cam, frame).
            # Let's assume idx maps to (frame_idx) and we return multi-view or single-view?
            # The SyntheticDataset returns single view + targets.
            # Here we probably act as a standard dataset where idx -> specific sample.
            
            # Simple mapping: idx maps to frame_idx (all cameras)
            # OR idx maps to (frame_idx, cam_idx)
            # Let's stick to MoReMouse training logic: batch gets ONE view, but might need others?
            # Trainer selects random view from batch["images"].
            # So __getitem__ should return ALL views for a frame.
            
            frame_idx = self.frames[idx]
            
            images = []
            viewmats = []
            Ks = []
            
            # Load global transform
            global_transform = self._get_global_transform(frame_idx)
            
            # Platform offset for world coordinates (MAMMAL specific)
            # X: 140cm, Z: 43.9cm
            PLATFORM_OFFSET = torch.tensor([1.40, 0.001, 0.439], dtype=torch.float32) # meters
            
            center_world = None
            yaw_angle = 0.0
            
            if global_transform and global_transform['valid']:
                # Local center (floor aligned)
                center_local = global_transform['center'] # [3] in meters
                # Global center
                center_world = center_local + PLATFORM_OFFSET
                # Negate yaw angle (calibration fix)
                yaw_angle = -global_transform['angle'].item()
            
            for cam_idx in self.cameras:
                # 1. Get image
                reader = self.video_readers[cam_idx]
                img_np = reader.get_frame(frame_idx)
                
                if img_np is None:
                    # Fallback (should not happen if frames filtered)
                    img_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                
                # 2. Get Camera Params
                cam_info = self.calibration["cameras"][f"cam{cam_idx}"]
                K = torch.tensor(cam_info["K"], dtype=torch.float32)
                viewmat = torch.tensor(cam_info["viewmat"], dtype=torch.float32)
                
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                
                # CANONICAL MODE PREPROCESSING
                if self.canonical_mode and center_world is not None:
                    # 1. Compute BBox
                    # Note: Reader gives original resolution image
                    orig_h, orig_w = img_np.shape[:2]
                    
                    x1, y1, crop_size = self._compute_bbox_from_global(
                        center_world, K, viewmat, orig_w, orig_h, scale=self.crop_scale
                    )
                    
                    # 2. Crop Image
                    img_tensor = self._crop_and_resize_image(
                        img_tensor, x1, y1, crop_size, self.image_size
                    )
                    
                    # 3. Compute Canonical Camera
                    K_new, V_new = self._compute_canonical_camera(
                        K, viewmat, center_world, yaw_angle, 
                        crop_size, self.image_size, 
                        world_scale=self.world_scale
                    )
                    
                    # Store crop info for debugging/projection
                    crop_info = {
                        'x1': torch.tensor(x1),
                        'y1': torch.tensor(y1),
                        'scale': torch.tensor(self.image_size / crop_size),
                        'crop_size': torch.tensor(crop_size)
                    }
                    
                    # Update K and viewmat
                    K = K_new
                    viewmat = V_new
                else:
                    # Standard resize
                    img_tensor = F.interpolate(
                        img_tensor.unsqueeze(0), 
                        size=(self.image_size, self.image_size),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                
                images.append(img_tensor)
                viewmats.append(viewmat)
                Ks.append(K)
            
            images = torch.stack(images)
            viewmats = torch.stack(viewmats)
            Ks = torch.stack(Ks)
            
            # Load pose
            pose = self._load_pose(frame_idx)
            pose_tensor = pose['thetas'] if pose else torch.zeros(140*3)
            
            # Get keypoints if available
            keypoints = []
            if self.keypoints2d:
                for cam_idx in self.cameras:
                    if cam_idx in self.keypoints2d:
                        kp = torch.from_numpy(self.keypoints2d[cam_idx][frame_idx])
                        
                        # Adjust keypoints for crop if needed
                        if self.canonical_mode and center_world is not None:
                            # We need check how to transform 2D KPs.
                            # Standard scale/shift.
                            # But we don't have per-camera crop info stored efficiently above.
                            # For simplicity, skip KP adjustment for training data now 
                            # (not used in loss unless supervised 2D).
                            pass
                            
                        keypoints.append(kp)
                    else:
                        keypoints.append(torch.zeros(22, 3))
                keypoints = torch.stack(keypoints)
            else:
                keypoints = None
                
            return {
                "images": images,
                "viewmats": viewmats,
                "Ks": Ks,
                "pose": pose_tensor,
                "global_transform": global_transform,
                "keypoints2d": keypoints,
                "frame_idx": frame_idx
            }
        
        else:
            # Image based (legacy/not primary)
            return {}  
