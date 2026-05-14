#!/usr/bin/env python3
"""
Dataset module for change detection.
Handles loading of pre/post image pairs, label remapping, EO/SAR fusion,
and data augmentation.
"""

import os
import warnings
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**31)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Suppress TIFF warnings
warnings.filterwarnings("ignore", message=".*TIFF.*")
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):
    """
    Dataset for binary change detection on satellite imagery.
    
    Expected directory structure (auto-detected):
        split/
        ├── A/          (pre-event images)
        ├── B/          (post-event images)
        └── OUT/        (ground truth masks)
    
    OR:
        split/
        ├── images1/    (pre-event images)
        ├── images2/    (post-event images)
        └── masks/      (ground truth masks)
    
    Label Remapping:
        0 (Background) -> 0 (No Change)
        1 (Intact)     -> 0 (No Change)
        2 (Damaged)    -> 1 (Change)
        3 (Destroyed)  -> 1 (Change)
    """

    # Possible directory naming conventions
    PRE_DIRS = ["pre-event", "A", "images1", "pre", "image1"]
    POST_DIRS = ["post-event", "B", "images2", "post", "image2"]
    MASK_DIRS = ["target", "OUT", "masks", "mask", "labels", "label"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 256,
        augment: bool = False,
        label_remap: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            root_dir: Root data directory containing split folders
            split: One of 'train', 'val', 'test'
            img_size: Target image size (square)
            augment: Whether to apply data augmentation
            label_remap: Label remapping dictionary
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.label_remap = label_remap or {0: 0, 1: 0, 2: 1, 3: 1}

        # Find directories
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.pre_dir = self._find_dir(self.PRE_DIRS, "pre-event images")
        self.post_dir = self._find_dir(self.POST_DIRS, "post-event images")
        self.mask_dir = self._find_dir(self.MASK_DIRS, "masks")

        # Get sorted file list (matched by name)
        self.filenames = self._get_matched_files()

        # Build augmentation pipeline
        self.transform = self._build_transforms()

    def _find_dir(self, candidates: List[str], description: str) -> Path:
        """Find the correct directory from a list of candidates."""
        for name in candidates:
            path = self.split_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find {description} directory in {self.split_dir}. "
            f"Tried: {candidates}"
        )

    def _get_matched_files(self) -> List[str]:
        """Get list of filenames present in all three directories."""
        pre_files = {f.stem: f.name for f in sorted(self.pre_dir.iterdir()) if f.is_file()}
        post_files = {f.stem: f.name for f in sorted(self.post_dir.iterdir()) if f.is_file()}
        mask_files = {f.stem: f.name for f in sorted(self.mask_dir.iterdir()) if f.is_file()}

        # Find common stems
        common_stems = sorted(set(pre_files.keys()) & set(post_files.keys()) & set(mask_files.keys()))

        if not common_stems:
            raise RuntimeError(
                f"No matching files found across directories.\n"
                f"  Pre ({self.pre_dir}): {len(pre_files)} files\n"
                f"  Post ({self.post_dir}): {len(post_files)} files\n"
                f"  Mask ({self.mask_dir}): {len(mask_files)} files"
            )

        self._pre_names = {stem: pre_files[stem] for stem in common_stems}
        self._post_names = {stem: post_files[stem] for stem in common_stems}
        self._mask_names = {stem: mask_files[stem] for stem in common_stems}

        return common_stems

    def _build_transforms(self) -> A.Compose:
        """Build augmentation pipeline using Albumentations."""
        transforms = []

        # Resize
        transforms.append(A.Resize(self.img_size, self.img_size))

        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3,
                ),
            ])

        # Normalize to [0, 1] - no ImageNet normalization since we have mixed sensors
        transforms.append(A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0))

        return A.Compose(
            transforms,
            additional_targets={"image2": "image"},
        )

    def _load_image(self, path: Path) -> np.ndarray:
        """Load an image, handling both RGB and grayscale."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to load image: {path}")

        # Convert BGR to RGB if 3-channel
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            # Grayscale - keep as 2D, albumentations handles it
            img = img[:, :, np.newaxis]  # Add channel dim -> (H, W, 1)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        return img.astype(np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and remap the ground truth mask."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {path}")

        # Apply label remapping
        remapped = np.zeros_like(mask, dtype=np.float32)
        for src_label, dst_label in self.label_remap.items():
            remapped[mask == src_label] = dst_label

        return remapped  # (H, W), values in {0, 1}

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.filenames[idx]

        # Load images and mask
        pre_img = self._load_image(self.pre_dir / self._pre_names[stem])
        post_img = self._load_image(self.post_dir / self._post_names[stem])
        mask = self._load_mask(self.mask_dir / self._mask_names[stem])

        # Ensure same spatial size before augmentation
        if pre_img.shape[:2] != post_img.shape[:2]:
            h = min(pre_img.shape[0], post_img.shape[0])
            w = min(pre_img.shape[1], post_img.shape[1])
            pre_img = cv2.resize(pre_img, (w, h))
            post_img = cv2.resize(post_img, (w, h))
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Ensure both images have same number of channels for augmentation
        # If they differ, convert both to 3-channel
        pre_channels = pre_img.shape[2] if len(pre_img.shape) == 3 else 1
        post_channels = post_img.shape[2] if len(post_img.shape) == 3 else 1

        if pre_channels == 1 and len(pre_img.shape) == 2:
            pre_img = pre_img[:, :, np.newaxis]
        if post_channels == 1 and len(post_img.shape) == 2:
            post_img = post_img[:, :, np.newaxis]

        # Apply augmentations (same transform to both images and mask)
        transformed = self.transform(
            image=pre_img,
            image2=post_img,
            mask=mask,
        )

        pre_tensor = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float()   # (C, H, W)
        post_tensor = torch.from_numpy(transformed["image2"]).permute(2, 0, 1).float()  # (C, H, W)
        mask_tensor = torch.from_numpy(transformed["mask"]).unsqueeze(0).float()         # (1, H, W)

        # Concatenate pre and post along channel dimension for model input
        # This handles EO (3+3=6ch), SAR (1+1=2ch), or mixed (3+1=4ch)
        fused = torch.cat([pre_tensor, post_tensor], dim=0)  # (C_pre+C_post, H, W)

        return {
            "image": fused,
            "mask": mask_tensor,
            "pre": pre_tensor,
            "post": post_tensor,
            "filename": stem,
        }

    def get_channel_info(self) -> Tuple[int, int, int]:
        """Get channel counts: (pre_channels, post_channels, total_channels)."""
        sample = self[0]
        pre_c = sample["pre"].shape[0]
        post_c = sample["post"].shape[0]
        return pre_c, post_c, pre_c + post_c

    def get_class_distribution(self) -> Dict[str, float]:
        """Compute class distribution across the dataset."""
        total_pixels = 0
        change_pixels = 0

        for idx in range(min(len(self), 200)):  # Sample for speed
            mask = self._load_mask(
                self.mask_dir / self._mask_names[self.filenames[idx]]
            )
            total_pixels += mask.size
            change_pixels += np.sum(mask > 0)

        no_change_ratio = 1.0 - (change_pixels / total_pixels)
        change_ratio = change_pixels / total_pixels

        return {
            "no_change": no_change_ratio,
            "change": change_ratio,
            "total_pixels_sampled": total_pixels,
            "samples_analyzed": min(len(self), 200),
        }


def get_dataloaders(config: dict) -> Tuple:
    """Create train, val, and test dataloaders from config."""
    from torch.utils.data import DataLoader

    label_remap = config.get("dataset", {}).get("label_remap", {0: 0, 1: 0, 2: 1, 3: 1})
    # Ensure keys are ints
    label_remap = {int(k): int(v) for k, v in label_remap.items()}

    img_size = config.get("dataset", {}).get("img_size", 256)
    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = config.get("training", {}).get("num_workers", 4)
    data_root = config.get("data", {}).get("root", "data")

    train_ds = ChangeDetectionDataset(
        root_dir=data_root, split="train",
        img_size=img_size, augment=True, label_remap=label_remap,
    )
    val_ds = ChangeDetectionDataset(
        root_dir=data_root, split="val",
        img_size=img_size, augment=False, label_remap=label_remap,
    )
    test_ds = ChangeDetectionDataset(
        root_dir=data_root, split="test",
        img_size=img_size, augment=False, label_remap=label_remap,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
