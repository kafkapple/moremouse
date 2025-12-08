"""
Dataset Classes for MoReMouse

Handles both synthetic (avatar-generated) and real captured data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class SyntheticDataset(Dataset):
    """
    Synthetic dataset generated from Gaussian Mouse Avatar.

    Each sample contains:
    - Input image (single view)
    - Multi-view target images
    - Camera parameters
    - Pose parameters
    - Geodesic embeddings

    Args:
        data_dir: Root directory of synthetic data
        split: "train" or "val"
        num_views: Number of views per sample
        image_size: Target image size
        transform: Optional transforms
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        num_views: int = 4,
        image_size: int = 378,
        transform=None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.transform = transform

        # Load sample list
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """Load sample metadata."""
        meta_file = self.data_dir / f"{self.split}_meta.json"

        if meta_file.exists():
            with open(meta_file, 'r') as f:
                samples = json.load(f)
        else:
            # Scan directory for samples
            samples = self._scan_directory()

        return samples

    def _scan_directory(self) -> List[Dict]:
        """Scan directory to find samples."""
        samples = []
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist. Using placeholder data.")
            return [{"frame_id": i} for i in range(100)]

        # Find all frame directories
        frame_dirs = sorted(split_dir.glob("frame_*"))

        for frame_dir in frame_dirs:
            frame_id = int(frame_dir.name.split("_")[1])
            samples.append({
                "frame_id": frame_id,
                "frame_dir": str(frame_dir),
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - input_image: [3, H, W] input view
            - target_images: [V, 3, H, W] target views
            - input_viewmat: [4, 4] input camera pose
            - target_viewmats: [V, 4, 4] target camera poses
            - K: [3, 3] camera intrinsics
            - pose: [J*3] joint angles
            - embedding: [H, W, 3] geodesic embedding (if available)
        """
        sample = self.samples[idx]

        # Placeholder: generate random data if not available
        if "frame_dir" not in sample:
            return self._generate_placeholder()

        frame_dir = Path(sample["frame_dir"])

        # Load input image
        input_path = frame_dir / "input.png"
        input_image = self._load_image(input_path)

        # Load target images
        target_images = []
        for v in range(self.num_views):
            target_path = frame_dir / f"view_{v:02d}.png"
            target_images.append(self._load_image(target_path))
        target_images = torch.stack(target_images)

        # Load camera parameters
        camera_file = frame_dir / "cameras.json"
        cameras = self._load_cameras(camera_file)

        # Load pose
        pose_file = frame_dir / "pose.npy"
        pose = self._load_pose(pose_file)

        # Load embedding if available
        embedding_path = frame_dir / "embedding.png"
        if embedding_path.exists():
            embedding = self._load_image(embedding_path)
        else:
            embedding = torch.zeros(3, self.image_size, self.image_size)

        # Apply transforms
        if self.transform is not None:
            input_image = self.transform(input_image)
            target_images = torch.stack([
                self.transform(t) for t in target_images
            ])

        return {
            "input_image": input_image,
            "target_images": target_images,
            "input_viewmat": cameras["input_viewmat"],
            "target_viewmats": cameras["target_viewmats"],
            "K": cameras["K"],
            "pose": pose,
            "embedding": embedding,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        if not path.exists():
            return torch.zeros(3, self.image_size, self.image_size)

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img

    def _load_cameras(self, path: Path) -> Dict[str, torch.Tensor]:
        """Load camera parameters."""
        if not path.exists():
            return {
                "input_viewmat": torch.eye(4),
                "target_viewmats": torch.eye(4).unsqueeze(0).expand(self.num_views, -1, -1),
                "K": self._default_intrinsics(),
            }

        with open(path, 'r') as f:
            data = json.load(f)

        return {
            "input_viewmat": torch.tensor(data["input_viewmat"]),
            "target_viewmats": torch.tensor(data["target_viewmats"]),
            "K": torch.tensor(data["K"]),
        }

    def _load_pose(self, path: Path) -> torch.Tensor:
        """Load pose parameters."""
        if not path.exists():
            return torch.zeros(140 * 3)

        pose = np.load(str(path))
        return torch.from_numpy(pose).float()

    def _default_intrinsics(self) -> torch.Tensor:
        """Get default camera intrinsics."""
        fov = 29.86  # degrees
        f = self.image_size / (2 * np.tan(np.radians(fov) / 2))

        K = torch.tensor([
            [f, 0, self.image_size / 2],
            [0, f, self.image_size / 2],
            [0, 0, 1],
        ], dtype=torch.float32)

        return K

    def _generate_placeholder(self) -> Dict[str, torch.Tensor]:
        """Generate placeholder data for testing."""
        return {
            "input_image": torch.randn(3, self.image_size, self.image_size),
            "target_images": torch.randn(self.num_views, 3, self.image_size, self.image_size),
            "input_viewmat": torch.eye(4),
            "target_viewmats": torch.eye(4).unsqueeze(0).expand(self.num_views, -1, -1).clone(),
            "K": self._default_intrinsics(),
            "pose": torch.zeros(140 * 3),
            "embedding": torch.zeros(3, self.image_size, self.image_size),
        }


class RealDataset(Dataset):
    """
    Real captured dataset for evaluation.

    Multi-camera setup with synchronized frames.

    Args:
        data_dir: Root directory of real data
        image_size: Target image size
        transform: Optional transforms
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        image_size: int = 378,
        transform=None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform

        # Load frame list
        self.frames = self._load_frames()

        # Load camera calibration
        self.cameras = self._load_calibration()

    def _load_frames(self) -> List[int]:
        """Load list of available frames."""
        frames = []
        frame_dir = self.data_dir / "frames"

        if frame_dir.exists():
            for f in sorted(frame_dir.glob("frame_*.png")):
                frame_id = int(f.stem.split("_")[1])
                frames.append(frame_id)
        else:
            frames = list(range(100))  # Placeholder

        return frames

    def _load_calibration(self) -> Dict:
        """Load camera calibration."""
        calib_file = self.data_dir / "calibration.json"

        if calib_file.exists():
            with open(calib_file, 'r') as f:
                return json.load(f)

        # Default calibration
        return {
            "num_cameras": 4,
            "intrinsics": self._default_intrinsics().numpy().tolist(),
            "extrinsics": [np.eye(4).tolist() for _ in range(4)],
        }

    def _default_intrinsics(self) -> torch.Tensor:
        """Get default camera intrinsics."""
        fov_h = 22.34  # degrees (horizontal)
        f = self.image_size / (2 * np.tan(np.radians(fov_h) / 2))

        K = torch.tensor([
            [f, 0, self.image_size / 2],
            [0, f, self.image_size / 2],
            [0, 0, 1],
        ], dtype=torch.float32)

        return K

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - images: [C, 3, H, W] images from all cameras
            - viewmats: [C, 4, 4] camera poses
            - K: [3, 3] camera intrinsics
        """
        frame_id = self.frames[idx]
        num_cameras = self.cameras["num_cameras"]

        # Load images from all cameras
        images = []
        for c in range(num_cameras):
            img_path = self.data_dir / "frames" / f"frame_{frame_id:06d}_cam{c}.png"
            img = self._load_image(img_path)
            images.append(img)
        images = torch.stack(images)

        # Camera parameters
        K = torch.tensor(self.cameras["intrinsics"], dtype=torch.float32)
        viewmats = torch.tensor(self.cameras["extrinsics"], dtype=torch.float32)

        # Apply transforms
        if self.transform is not None:
            images = torch.stack([self.transform(img) for img in images])

        return {
            "images": images,
            "viewmats": viewmats,
            "K": K,
            "frame_id": frame_id,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        if not path.exists():
            return torch.zeros(3, self.image_size, self.image_size)

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img
