"""
SynthWave Training Pipeline
Complete training script with data loading, augmentation, and evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from model_architecture import DeepfakeDetector, print_model_summary


class DeepfakeTrainer:
    """Handles training and evaluation of deepfake detection models"""
    
    def __init__(self, data_dir='data/processed', model_type='custom', 
                 img_size=(224, 224), batch_size=32):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing train/val/test splits
            model_type: 'custom', 'mobilenet', or 'efficientnet'
            img_size: Input image size
            batch_size: Batch size for training
        """
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Create output directory
        self.output_dir = Path('outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        # Class names
        self.class_names = ['fake', 'real']
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation/Test data (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\n✓ Data generators created:")
        print(f"  Training samples: {self.train_generator.samples}")
        print(f"  Validation samples: {self.val_generator.samples}")
        print(f"  Test samples: {self.test_generator.samples}")
        print(f"  Classes: {self.train_generator.class_indices}")
    
    def build_model(self):
        """Build model based on specified type"""
        
        print(f"\n🏗️ Building {self.model_type} model...")
        
        input_shape = (*self.img_size, 3)
        
        if self.model_type == 'custom':
            self.model = DeepfakeDetector.create_custom_cnn(input_shape=input_shape)
        elif self.model_type == 'mobilenet':
            self.model = DeepfakeDetector.create_mobilenet_model(input_shape=input_shape)
        elif self.model_type == 'efficientnet':
            self.model = DeepfakeDetector.create_efficientnet_model(input_shape=input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print_model_summary(self.model)
    
    def train(self, epochs=50, patience=10):
        """
        Train the model
        
        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        
        print("\n🚀 Starting training...")
        
        # Callbacks
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / f'best_model_{self.model_type}.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                save_format='keras',  # Use native Keras format
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=str(self.output_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        # Calculate steps
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.val_generator.samples // self.batch_size
        
        # Train
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training complete!")
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'SynthWave Training History - {self.model_type.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_history_{self.model_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Training history plot saved to {self.output_dir}")
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        
        print("\n📊 Evaluating model on test set...")
        
        # Get predictions
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      digits=4)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {self.model_type.upper()}', 
                 fontweight='bold', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{self.model_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, predictions[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_type.upper()}', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'roc_curve_{self.model_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            self.test_generator, verbose=0
        )
        
        metrics = {
            'model_type': self.model_type,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        with open(self.output_dir / f'metrics_{self.model_type}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n✓ Evaluation complete! Metrics saved to {self.output_dir}")
        
        return metrics


def main():
    """Main training function"""
    
    print("\n" + "="*70)
    print("🌊 SYNTHWAVE DEEPFAKE DETECTOR - TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    MODEL_TYPE = 'custom'  # Options: 'custom', 'mobilenet', 'efficientnet'
    EPOCHS = 50
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    
    # Initialize trainer
    trainer = DeepfakeTrainer(
        data_dir='data/processed',
        model_type=MODEL_TYPE,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Create data generators
    trainer.create_data_generators()
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train(epochs=EPOCHS, patience=10)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    metrics = trainer.evaluate_model()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()