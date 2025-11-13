"""
Download a small sample dataset for quick testing
This uses the 140k Real and Fake Faces dataset from Kaggle
"""

import os
from pathlib import Path

# ✅ Must come BEFORE importing Kaggle API
os.environ['KAGGLE_CONFIG_DIR'] = r"A:\Projects\FauxFly"  # folder containing kaggle.json

from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset="uditsharma72/real-vs-fake-faces", output_dir="data/raw"):
    """
    Download and extract a dataset from Kaggle using Kaggle API.
    Looks for kaggle.json in the current folder or default ~/.kaggle/
    """
    os.makedirs(output_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    
    print(f"📦 Downloading {dataset} to {output_dir} ...")
    api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print("✅ Download complete!")


# def download_sample_dataset():
#     """
#     Provide instructions and create necessary directories
#     """
#     print("="*60)
#     print("SAMPLE DATASET DOWNLOAD INSTRUCTIONS")
#     print("="*60)
    
#     print("\nOption 1: Kaggle Dataset (140K Real and Fake Faces)")
#     print("-" * 60)
#     print("1. Install Kaggle API: pip install kaggle")
#     print("2. Place kaggle.json in project folder or ~/.kaggle/")
#     print("3. Run: python this_script.py to download automatically")
#     print("4. Dataset: xhlulu/140k-real-and-fake-faces")
    
#     print("\nOption 2: Manual Collection")
#     print("-" * 60)
#     print("REAL FACES: CelebA, FFHQ, VGGFace2")
#     print("FAKE FACES: ThisPersonDoesNotExist, StyleGAN, FaceForensics++")

#     print("\n" + "="*60)
#     print("After downloading, organize your data as:")
#     print("  data/raw/real/  <- Place all real face images here")
#     print("  data/raw/fake/  <- Place all fake face images here")
#     print("="*60)
    
#     Path('data/raw/real').mkdir(parents=True, exist_ok=True)
#     Path('data/raw/fake').mkdir(parents=True, exist_ok=True)
    
#     print("\n✓ Directories created. You can now download with `download_kaggle_dataset()`")


if __name__ == "__main__":
    # Uncomment this line to automatically download the dataset:
    # download_kaggle_dataset()
    download_kaggle_dataset()
