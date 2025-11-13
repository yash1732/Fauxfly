"""
SynthWave Dataset Preparation Script
Handles face detection, cropping, and dataset splitting using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import argparse

class FaceCropper:
    """
    Detects and crops faces from images using MediaPipe.
    """
    def __init__(self, confidence=0.5):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=confidence
        )

    def detect_and_crop_face(self, image_path, output_size=(224, 224)):
        """
        Detect the largest face in an image and crop it.
        
        Args:
            image_path: Path to the input image.
            output_size: The target size for the cropped face.
            
        Returns:
            Cropped face image or None if no face detected.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.face_detection.process(img_rgb)
            
            if not results.detections:
                return None

            # Find the largest face
            largest_detection = max(results.detections, 
                                    key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            
            bbox = largest_detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            
            # Get absolute coordinates
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            # Add padding (20%)
            padding = int(0.2 * max(x2 - x1, y2 - y1))
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face = img[y1:y2, x1:x2]
            
            if face.size == 0:
                return None
                
            # Resize
            face_resized = cv2.resize(face, output_size, interpolation=cv2.INTER_AREA)
            return face_resized
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

def process_and_split(raw_dir, processed_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Find all images, crop faces, and split into train/val/test sets.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Setup output directories
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (processed_path / split / label).mkdir(parents=True, exist_ok=True)
            
    print("✓ Output directories created.")
    
    cropper = FaceCropper()
    
    # Process each label
    for label in ['real', 'fake']:
        print(f"\nProcessing '{label}' images...")
        
        # Recursively find all images for this label
        # This is robust to subfolders
        image_files = []
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        # Search for any path containing the label
        for ext in image_extensions:
            image_files.extend(raw_path.rglob(f"**/{label}*/**/*{ext}")) # e.g., .../train/real/...
            image_files.extend(raw_path.rglob(f"**/{label}*/*{ext}")) # e.g., .../real/...

        # Handle datasets that just use '0' for fake and '1' for real
        if label == 'fake':
            for ext in image_extensions:
                image_files.extend(raw_path.rglob(f"**/0/*{ext}")) # '0' often means fake
        if label == 'real':
            for ext in image_extensions:
                image_files.extend(raw_path.rglob(f"**/1/*{ext}")) # '1' often means real

        image_files = sorted(list(set(image_files))) # Remove duplicates
        
        print(f"  Found {len(image_files)} raw images.")
        
        if not image_files:
            print(f"  Warning: No images found for label '{label}' in {raw_path}.")
            print("  Please check your download folder and dataset structure.")
            continue
            
        random.shuffle(image_files)
        
        # Define split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        split_map = {
            'train': image_files[:n_train],
            'val': image_files[n_train : n_train + n_val],
            'test': image_files[n_train + n_val:]
        }
        
        # Process and save
        total_saved = 0
        for split, files in split_map.items():
            print(f"  Processing {len(files)} images for '{split}' split...")
            saved_count = 0
            
            output_dir = processed_path / split / label
            
            for img_path in tqdm(files, desc=f"  {split}/{label}"):
                face = cropper.detect_and_crop_face(img_path)
                
                if face is not None:
                    output_name = f"{img_path.stem}_cropped.jpg"
                    output_path = output_dir / output_name
                    cv2.imwrite(str(output_path), face)
                    saved_count += 1
            
            print(f"  ✓ {split}/{label}: {saved_count}/{len(files)} faces saved.")
            total_saved += saved_count
            
        print(f"  Total '{label}' faces saved: {total_saved}")

def verify_dataset(processed_dir):
    """Print dataset statistics"""
    print("\n" + "="*50)
    print("PROCESSED DATASET STATISTICS")
    print("="*50)
    
    processed_path = Path(processed_dir)
    
    for split in ['train', 'val', 'test']:
        split_path = processed_path / split
        
        if not split_path.exists():
            continue
        
        real_count = len(list((split_path / 'real').glob('*.jpg')))
        fake_count = len(list((split_path / 'fake').glob('*.jpg')))
        total = real_count + fake_count
        
        print(f"\n{split.upper()}:")
        print(f"  Real: {real_count}")
        print(f"  Fake: {fake_count}")
        print(f"  Total: {total}")
        if total > 0:
            print(f"  Balance: {real_count/total*100:.1f}% real, {fake_count/total*100:.1f}% fake")
        else:
            print("  (No images)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare deepfake dataset with face cropping.")
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing the raw downloaded dataset."
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to save the processed (cropped, split) dataset."
    )
    
    args = parser.parse_args()

    print("\n" + "="*50)
    print("SYNTHWAVE DATASET PREPARATION (MediaPipe Edition)")
    print("="*50)
    
    process_and_split(args.raw_dir, args.processed_dir)
    verify_dataset(args.processed_dir)
    
    print("\n✓ Dataset preparation complete!")
    print("You can now proceed to training.")