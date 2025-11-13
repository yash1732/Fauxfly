"""
SynthWave Dataset Preparation Script
Handles downloading, face detection, cropping, and organizing datasets
"""

import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import shutil
from pathlib import Path

class DatasetPreparator:
    def __init__(self, base_dir='data'):
        """
        Initialize dataset preparator
        
        Args:
            base_dir: Root directory for dataset storage
        """
        self.base_dir = Path(base_dir)
        self.detector = MTCNN()
        
        # Create directory structure
        self.setup_directories()
    
    def setup_directories(self):
        """Create the required folder structure"""
        directories = [
            'raw/real',
            'raw/fake',
            'processed/train/real',
            'processed/train/fake',
            'processed/test/real',
            'processed/test/fake',
            'processed/val/real',
            'processed/val/fake'
        ]
        
        for dir_path in directories:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created successfully!")
    
    def detect_and_crop_face(self, image_path, output_size=(224, 224)):
        """
        Detect face in image and crop it
        
        Args:
            image_path: Path to input image
            output_size: Size of output cropped face
            
        Returns:
            Cropped face image or None if no face detected
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(img_rgb)
            
            if len(detections) == 0:
                return None
            
            # Get the largest face (by bounding box area)
            largest_face = max(detections, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = largest_face['box']
            
            # Add padding (20% on each side)
            padding = int(0.2 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            # Crop face
            face = img[y1:y2, x1:x2]
            
            # Resize to standard size
            face_resized = cv2.resize(face, output_size, interpolation=cv2.INTER_AREA)
            
            return face_resized
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def process_dataset(self, source_dir, label, split='train', train_ratio=0.7, val_ratio=0.15):
        """
        Process all images in a directory
        
        Args:
            source_dir: Directory containing raw images
            label: 'real' or 'fake'
            split: Which split to process ('train', 'test', 'val', or 'auto' for automatic splitting)
            train_ratio: Ratio of images for training (if split='auto')
            val_ratio: Ratio of images for validation (if split='auto')
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"Warning: {source_path} does not exist!")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\nProcessing {len(image_files)} {label} images from {source_dir}...")
        
        # Auto-split if requested
        if split == 'auto':
            np.random.shuffle(image_files)
            n_train = int(len(image_files) * train_ratio)
            n_val = int(len(image_files) * val_ratio)
            
            splits = {
                'train': image_files[:n_train],
                'val': image_files[n_train:n_train + n_val],
                'test': image_files[n_train + n_val:]
            }
        else:
            splits = {split: image_files}
        
        # Process each split
        for split_name, files in splits.items():
            output_dir = self.base_dir / 'processed' / split_name / label
            successful = 0
            
            for img_path in tqdm(files, desc=f"{split_name}/{label}"):
                face = self.detect_and_crop_face(img_path)
                
                if face is not None:
                    output_path = output_dir / f"{img_path.stem}_cropped.jpg"
                    cv2.imwrite(str(output_path), face)
                    successful += 1
            
            print(f"✓ {split_name}/{label}: {successful}/{len(files)} faces detected and saved")
    
    def augment_dataset(self, input_dir, augment_factor=2):
        """
        Apply data augmentation to increase dataset size
        
        Args:
            input_dir: Directory containing images to augment
            augment_factor: Number of augmented versions per image
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            return
        
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        print(f"\nAugmenting {len(image_files)} images in {input_dir}...")
        
        for img_path in tqdm(image_files, desc="Augmenting"):
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            for i in range(augment_factor):
                # Apply random transformations
                augmented = img.copy()
                
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    augmented = cv2.flip(augmented, 1)
                
                # Random brightness adjustment
                brightness = np.random.uniform(0.7, 1.3)
                augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
                
                # Random rotation (-10 to +10 degrees)
                angle = np.random.uniform(-10, 10)
                h, w = augmented.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                augmented = cv2.warpAffine(augmented, M, (w, h))
                
                # Save augmented image
                output_path = input_path / f"{img_path.stem}_aug{i}.jpg"
                cv2.imwrite(str(output_path), augmented)
        
        print(f"✓ Augmentation complete!")

    def verify_dataset(self):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        for split in ['train', 'val', 'test']:
            split_path = self.base_dir / 'processed' / split
            
            if not split_path.exists():
                continue
            
            real_count = len(list((split_path / 'real').glob('*.jpg')))
            fake_count = len(list((split_path / 'fake').glob('*.jpg')))
            total = real_count + fake_count
            
            print(f"\n{split.upper()}:")
            print(f"  Real: {real_count}")
            print(f"  Fake: {fake_count}")
            print(f"  Total: {total}")
            print(f"  Balance: {real_count/(total+0.001)*100:.1f}% real, {fake_count/(total+0.001)*100:.1f}% fake")


# Example usage
if __name__ == "__main__":
    # Initialize preparator
    prep = DatasetPreparator(base_dir='data')
    
    # Option 1: Process pre-organized datasets
    # Place your raw images in data/raw/real and data/raw/fake
    # Then run automatic splitting:
    
    print("\n" + "="*50)
    print("SYNTHWAVE DATASET PREPARATION")
    print("="*50)
    
    # Process real images with auto-split
    prep.process_dataset(
        source_dir='data/raw/real',
        label='real',
        split='auto',
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Process fake images with auto-split
    prep.process_dataset(
        source_dir='data/raw/fake',
        label='fake',
        split='auto',
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Optional: Augment training data
    # prep.augment_dataset('data/processed/train/real', augment_factor=2)
    # prep.augment_dataset('data/processed/train/fake', augment_factor=2)
    
    # Verify final dataset
    prep.verify_dataset()
    
    print("\n✓ Dataset preparation complete!")
    print("You can now proceed to training.")