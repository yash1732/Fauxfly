"""
SynthWave Grad-CAM Implementation
Provides visual explainability for model predictions
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) implementation"""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to visualize.
                       If None, uses the last conv layer.
        """
        self.model = model
        self.layer_name = layer_name
        
        # Find the last convolutional layer if not specified
        if self.layer_name is None:
            for layer in reversed(model.layers):
                # Check if layer is a Conv2D layer (compatible with Keras 3.x)
                if hasattr(layer, 'output') and layer.__class__.__name__ in ['Conv2D', 'SeparableConv2D', 'DepthwiseConv2D']:
                    self.layer_name = layer.name
                    break
                # Fallback: check output shape if available
                elif hasattr(layer, 'output'):
                    try:
                        output_shape = layer.output.shape
                        if len(output_shape) == 4:  # Conv layer has 4D output
                            self.layer_name = layer.name
                            break
                    except:
                        continue
        
        if self.layer_name is None:
            raise ValueError("No convolutional layer found in the model. Please specify layer_name manually.")
        
        print(f"Using layer '{self.layer_name}' for Grad-CAM visualization")
        
        # Create gradient model
        self.grad_model = keras.models.Model(
            inputs=[model.input],
            outputs=[model.get_layer(self.layer_name).output, model.output]
        )
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed, shape: [H, W, 3])
            class_idx: Target class index. If None, uses predicted class.
            eps: Small value to avoid division by zero
            
        Returns:
            heatmap: Grad-CAM heatmap (normalized to [0, 1])
        """
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get conv layer output and model predictions
            conv_outputs, predictions = self.grad_model(image_batch)
            
            # If class_idx not specified, use predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of class score with respect to conv output
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute guided gradients (ReLU applied)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv output by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to the heatmap
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize heatmap to [0, 1]
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            image: Original image (RGB, [0, 255])
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            superimposed: Image with heatmap overlay
        """
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed
    
    def explain_prediction(self, image_path, model, preprocess_fn=None, 
                          class_names=None, save_path=None):
        """
        Generate complete explanation for a single image
        
        Args:
            image_path: Path to input image
            model: Trained model
            preprocess_fn: Function to preprocess image
            class_names: List of class names
            save_path: Path to save visualization
            
        Returns:
            prediction: Model prediction
            confidence: Prediction confidence
            heatmap: Grad-CAM heatmap
        """
        
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for model input
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Preprocess
        if preprocess_fn is not None:
            img_processed = preprocess_fn(img_resized)
        else:
            img_processed = img_resized / 255.0
        
        # Get prediction
        pred_batch = np.expand_dims(img_processed, axis=0)
        predictions = model.predict(pred_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        
        # Generate heatmap
        heatmap = self.compute_heatmap(img_processed, class_idx)
        
        # Overlay heatmap
        superimposed = self.overlay_heatmap(heatmap, img_resized)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_resized)
        axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontweight='bold', fontsize=12)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(superimposed)
        if class_names:
            pred_label = class_names[class_idx]
        else:
            pred_label = f"Class {class_idx}"
        axes[2].set_title(f'Prediction: {pred_label}\nConfidence: {confidence:.2f}%',
                         fontweight='bold', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Explanation saved to {save_path}")
        
        plt.show()
        
        return class_idx, confidence, heatmap


def batch_explain(model, image_dir, output_dir, num_samples=10, class_names=None):
    """
    Generate Grad-CAM explanations for multiple images
    
    Args:
        model: Trained model
        image_dir: Directory containing test images
        output_dir: Directory to save explanations
        num_samples: Number of samples to explain
        class_names: List of class names
    """
    from pathlib import Path
    import random
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all images
    image_files = list(image_dir.rglob('*.jpg')) + list(image_dir.rglob('*.png'))
    
    # Sample random images
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create Grad-CAM object
    gradcam = GradCAM(model)
    
    print(f"\n📊 Generating Grad-CAM explanations for {len(sampled_images)} images...")
    
    for i, img_path in enumerate(sampled_images):
        save_path = output_dir / f'gradcam_explanation_{i+1}.png'
        
        try:
            gradcam.explain_prediction(
                image_path=img_path,
                model=model,
                class_names=class_names,
                save_path=save_path
            )
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"\n✓ All explanations saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    print("\n🔍 SynthWave Grad-CAM Explainability Demo\n")
    
    # Load trained model
    model_path = 'outputs/best_model_custom.keras'
    
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Class names
        class_names = ['Fake', 'Real']
        
        # Generate explanations for test images
        batch_explain(
            model=model,
            image_dir='data/processed/test',
            output_dir='outputs/gradcam',
            num_samples=10,
            class_names=class_names
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure you have trained the model first (run 03_train_model.py)")