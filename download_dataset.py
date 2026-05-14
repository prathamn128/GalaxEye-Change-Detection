#!/usr/bin/env python3
"""
Download the change detection dataset from HuggingFace.
Downloads train.zip, val.zip, test.zip and extracts them to data/.
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path


# HuggingFace direct download URLs
BASE_URL = "https://huggingface.co/datasets/doron333/change-detection-dataset/resolve/main"
FILES = {
    "train.zip": f"{BASE_URL}/train.zip",
    "val.zip": f"{BASE_URL}/val.zip",
    "test.zip": f"{BASE_URL}/test.zip",
}

DATA_DIR = Path("data")


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists.")
        return

    print(f"  Downloading {dest.name} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    progress = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"  {dest.name}",
        ncols=80,
    )

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()
    print(f"  [DONE] {dest.name} ({total_size / 1e9:.2f} GB)")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    folder_name = zip_path.stem  # train, val, test
    target = extract_to / folder_name

    if target.exists() and any(target.iterdir()):
        print(f"  [SKIP] {folder_name}/ already extracted.")
        return

    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  [DONE] Extracted to {target}/")


def verify_dataset() -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"  [WARN] {split}/ not found!")
            continue

        # Count samples
        images_dir = None
        masks_dir = None

        # Try different possible structures
        for img_name in ["images", "image", "pre", "A"]:
            candidate = split_dir / img_name
            if candidate.exists():
                images_dir = candidate
                break

        for mask_name in ["masks", "mask", "label", "labels", "OUT"]:
            candidate = split_dir / mask_name
            if candidate.exists():
                masks_dir = candidate
                break

        if images_dir:
            n_images = len(list(images_dir.glob("*")))
        else:
            # List subdirectories to understand structure
            subdirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
            n_images = "?"
            print(f"  {split}/: subdirs = {subdirs}")

        print(f"  {split}/: {n_images} samples" if images_dir else "")


def main():
    print("=" * 60)
    print("GalaxEye Change Detection - Dataset Downloader")
    print("=" * 60)

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_dir = DATA_DIR / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    # Download
    print("\n[1/3] Downloading dataset files...")
    for filename, url in FILES.items():
        zip_path = zip_dir / filename
        download_file(url, zip_path)

    # Extract
    print("\n[2/3] Extracting archives...")
    for filename in FILES:
        zip_path = zip_dir / filename
        if zip_path.exists():
            extract_zip(zip_path, DATA_DIR)

    # Verify
    print("\n[3/3] Verifying dataset...")
    verify_dataset()

    # Show directory structure
    print("\n[INFO] Dataset directory structure:")
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            subdirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            files_count = len([f for f in split_dir.iterdir() if f.is_file()])
            print(f"  {split}/")
            for sd in subdirs:
                sd_path = split_dir / sd
                n = len(list(sd_path.glob("*")))
                print(f"    {sd}/ ({n} files)")
            if files_count:
                print(f"    ({files_count} files)")

    print("\n[SUCCESS] Dataset ready!")


if __name__ == "__main__":
    main()
