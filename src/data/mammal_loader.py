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
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self._explicit_pose_dir = Path(pose_dir) if pose_dir else None

        # Detect data format (video or image)
        self.use_video = self._detect_video_format()

        if self.use_video:
            # Video-based loading
            self.video_readers = self._init_video_readers()
            self.pose_dir = self._explicit_pose_dir or self._find_pose_dir()
            self.calibration = self._load_calibration()
            self.global_transform = self._load_global_transform()  # center_rotation.npz

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
                if num_frames is not None:
                    self.frames = self.frames[:num_frames]
            else:
                self.frames = []

            print(f"MAMMAL Dataset (video mode): {len(self.frames)} frames, {len(self.cameras)} cameras")
        else:
            # Image-based loading
            self.video_readers = {}
            self.image_dir = self._find_image_dir()
            self.pose_dir = self._explicit_pose_dir or self._find_pose_dir()
            self.calibration = self._load_calibration()
            self.global_transform = self._load_global_transform()

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
        """
        calib = {"cameras": {}}

        for i, cam in enumerate(cam_list):
            K = np.array(cam['K'])
            R = np.array(cam['R'])
            T = np.array(cam['T'])

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
            'center': torch.from_numpy(center).float(),  # (3,)
            'angle': torch.tensor(angle, dtype=torch.float32),  # scalar
        }

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

            # Get camera params
            cam_params = self.get_camera_params(cam_idx)
            viewmats.append(np.array(cam_params["viewmat"]))
            Ks.append(np.array(cam_params["K"]))

        # Load pose (return empty tensor if not available)
        pose_data = self._load_pose(frame_idx)
        has_pose = pose_data is not None

        if has_pose and 'thetas' in pose_data:
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

                if isinstance(R_global, np.ndarray):
                    R_global = torch.from_numpy(R_global).float()
                    T_global = torch.from_numpy(T_global).float()
                    scale = torch.tensor(scale, dtype=torch.float32)

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

        return {
            "images": torch.from_numpy(np.stack(images)),  # [C, H, W, 3]
            "viewmats": torch.from_numpy(np.stack(viewmats)).float(),  # [C, 4, 4]
            "Ks": torch.from_numpy(np.stack(Ks)).float(),  # [C, 3, 3]
            "pose": pose_tensor,  # [140*3] flattened axis-angle rotations
            "has_pose": has_pose,  # Flag to indicate if pose was loaded
            "mammal_global": mammal_global,  # Dict with R, T, s from MAMMAL fitting
            "global_transform": global_trans,  # Dict with center, angle from center_rotation.npz
            "frame_idx": frame_idx,
        }


def create_mammal_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 1,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create MAMMAL multi-view dataloader."""
    dataset = MAMMALMultiviewDataset(data_dir, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
