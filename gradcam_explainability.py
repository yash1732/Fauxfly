# YS
"""
SynthWave Grad-CAM Implementation (Complete Rewrite)
Provides visual explainability for model predictions using a robust,
functional approach based on the official Keras.io tutorial.
This version uses the correct "two-model" approach to guarantee
the gradient chain is not broken.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def find_last_conv_layer_name(model):
    """
    Finds the name of the last convolutional or activation layer
    before the global pooling. This is crucial for Grad-CAM.
    """
    
    # For custom model, 'relu4' is the last activation after conv.
    # This is a much better layer for gradients than 'conv4'.
    for layer in reversed(model.layers):
        if layer.name == 'relu4':
            print(f"✓ Found 'relu4' (ideal for custom model).")
            return 'relu4'

    # Fallback for transfer learning models (MobileNet, EfficientNet)
    # Find the last layer *before* GlobalAveragePooling2D
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            continue # Skip this layer, we want the one *before* it

        # Return the last activation or conv layer we find
        if (isinstance(layer, keras.layers.Activation) or 
            isinstance(layer, keras.layers.Conv2D)):
            print(f"✓ Found last conv/activation layer: '{layer.name}'")
            return layer.name
    
    # If all else fails
    raise ValueError("Could not find a suitable layer for Grad-CAM.")


def make_gradcam_heatmap(image_batch, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap using the robust "two-model" method.
    
    Args:
        image_batch: A preprocessed image batch (shape: [1, H, W, 3])
        model: The trained Keras model.
        last_conv_layer_name: The name of the conv layer to use.
        pred_index: (Optional) The class index. If None, uses max prediction.
        
    Returns:
        The Grad-CAM heatmap.
    """
    
    # Cast image_batch to float32
    image_batch = tf.cast(image_batch, tf.float32)
    
    # 1. Create the "conv model" that outputs the activation map
    conv_model = keras.models.Model(
        model.inputs, model.get_layer(last_conv_layer_name).output
    )

    # 2. Create the "classifier model" that takes the activation map as input
    classifier_input = keras.Input(shape=model.get_layer(last_conv_layer_name).output.shape[1:])
    
    x = classifier_input
    # Find the index of the last conv layer
    layer_index = -1
    for i, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            layer_index = i
            break
    
    if layer_index == -1:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

    # Add all subsequent layers to the new classifier model
    for layer in model.layers[layer_index + 1:]:
        x = layer(x)
    
    classifier_model = keras.models.Model(classifier_input, x)

    # 3. Run the GradientTape
    with tf.GradientTape() as tape:
        # Get the output of the conv model
        # We pass image_batch as a list, as model.inputs is a list
        conv_outputs = conv_model([image_batch])
        # Tell the tape to "watch" this tensor, as it's the source
        tape.watch(conv_outputs)
        # Get the predictions *from* the conv outputs
        # This creates the mathematical link: preds = f(conv_outputs)
        preds = classifier_model(conv_outputs)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 4. Get the gradient
    # This will now work, as class_channel is a function of conv_outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    if grads is None:
        print("!! Warning: Gradient was None. This can happen. Returning empty heatmap.")
        return None # Handle the None case gracefully

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization, we will normalize the heatmap between 0 & 1
    # and apply ReLU
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
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


def explain_prediction(model, last_conv_layer_name, image_path, class_names=None, save_path=None):
    """
    Generate complete explanation for a single image
    
    Args:
        model: Trained model
        last_conv_layer_name: Name of the conv layer
        image_path: Path to input image
        class_names: List of class names
        save_path: Path to save visualization
    """
    
    # Load and preprocess image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"!! Error: Could not read image {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize for model input
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Preprocess
    img_processed = img_resized / 255.0
    img_batch = np.expand_dims(img_processed, axis=0)

    # Get prediction
    predictions = model(img_batch, training=False)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name, class_idx)
    
    if heatmap is None:
        print(f"Skipping visualization for {image_path} due to gradient error.")
        return

    # Overlay heatmap
    superimposed = overlay_heatmap(heatmap, img_resized)
    
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
    
    plt.show(block=False) # Use non-blocking show
    plt.close(fig) # Close the figure to free memory

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
    
    if not image_files:
        print(f"!! Error: No images found in {image_dir}")
        return

    # Sample random images
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Find the conv layer ONCE
    try:
        last_conv_layer_name = find_last_conv_layer_name(model)
        print(f"Using layer '{last_conv_layer_name}' for Grad-CAM.")
    except Exception as e:
        print(f"!! FATAL: {e}")
        return

    print(f"\n📊 Generating Grad-CAM explanations for {len(sampled_images)} images...")
    
    for i, img_path in enumerate(sampled_images):
        save_path = output_dir / f'gradcam_explanation_{i+1}.png'
        
        try:
            explain_prediction(
                model=model,
                last_conv_layer_name=last_conv_layer_name,
                image_path=str(img_path), # Ensure path is a string
                class_names=class_names,
                save_path=save_path
            )
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("A fatal error occurred in Grad-CAM. Aborting.")
            break 
    
    print(f"\n✓ All explanations saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    print("\n🔍 SynthWave Grad-CAM Explainability Demo\n")
    
    # Load trained model
    model_path = 'outputs/best_model_custom.keras'
    
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Build the model on load
        if not model.built:
             model(tf.zeros((1, 224, 224, 3)))
        
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
        import traceback
        traceback.print_exc()
        print("Please ensure you have trained the model first (run 03_train_model.py)")