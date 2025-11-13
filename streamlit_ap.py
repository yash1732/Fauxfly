"""
SynthWave Streamlit Web Application
Interactive interface for deepfake detection with Grad-CAM visualization
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import io

# Import Grad-CAM
import sys
sys.path.append('.')
from gradcam_explainability import GradCAM


# Page configuration
st.set_page_config(
    page_title="SynthWave Deepfake Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fake-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .real-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load trained model with caching"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Resize
    img_resized = cv2.resize(image, target_size)
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized


def predict_image(model, image, gradcam):
    """Make prediction and generate Grad-CAM"""
    
    # Preprocess
    img_batch, img_resized = preprocess_image(image)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    # Generate Grad-CAM heatmap
    img_normalized = img_batch[0]
    heatmap = gradcam.compute_heatmap(img_normalized, class_idx)
    
    # Overlay heatmap
    superimposed = gradcam.overlay_heatmap(heatmap, img_resized)
    
    return class_idx, confidence, heatmap, superimposed, img_resized


def display_results(class_idx, confidence, img_resized, heatmap, superimposed):
    """Display prediction results with Grad-CAM"""
    
    class_names = ['Fake', 'Real']
    label = class_names[class_idx]
    
    # Result header
    if label == 'Fake':
        st.markdown(f"""
            <div class="result-box fake-result">
                <h2>🚨 FAKE DETECTED</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box real-result">
                <h2>✅ REAL IMAGE</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    
    # Display images in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_resized, caption='Original Image', use_container_width=True)
    
    with col2:
        # Convert heatmap to displayable format
        heatmap_display = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, caption='Grad-CAM Heatmap', use_container_width=True)
    
    with col3:
        st.image(superimposed, caption='Explainability Overlay', use_container_width=True)
    
    # Confidence meter
    st.markdown("### Prediction Confidence")
    st.progress(confidence / 100)
    
    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Prediction</h4>
                <h2>{label}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Confidence Score</h4>
                <h2>{confidence:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)


def process_video(model, gradcam, video_file, max_frames=30):
    """Process video file frame by frame"""
    
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    st.info(f"Video Info: {total_frames} frames @ {fps} FPS")
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    predictions = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Predict
        try:
            class_idx, confidence, _, _, _ = predict_image(model, frame_rgb, gradcam)
            predictions.append({
                'frame': frame_idx,
                'class': class_idx,
                'confidence': confidence
            })
        except:
            pass
        
        # Update progress
        progress = (i + 1) / len(frame_indices)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {i+1}/{len(frame_indices)}")
    
    cap.release()
    
    # Aggregate results
    if predictions:
        fake_count = sum(1 for p in predictions if p['class'] == 0)
        real_count = len(predictions) - fake_count
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        fake_percentage = (fake_count / len(predictions)) * 100
        
        # Display results
        st.markdown("### 📊 Video Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Frames Analyzed", len(predictions))
        
        with col2:
            st.metric("Fake Frames", f"{fake_count} ({fake_percentage:.1f}%)")
        
        with col3:
            st.metric("Average Confidence", f"{avg_confidence:.2f}%")
        
        # Overall verdict
        if fake_percentage > 50:
            st.error(f"🚨 Video is likely FAKE ({fake_percentage:.1f}% fake frames)")
        else:
            st.success(f"✅ Video is likely REAL ({100-fake_percentage:.1f}% real frames)")
        
        # Plot frame-by-frame analysis
        fig, ax = plt.subplots(figsize=(12, 4))
        frames = [p['frame'] for p in predictions]
        classes = [p['class'] for p in predictions]
        colors = ['red' if c == 0 else 'green' for c in classes]
        
        ax.scatter(frames, classes, c=colors, alpha=0.6, s=100)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Prediction (0=Fake, 1=Real)')
        ax.set_title('Frame-by-Frame Analysis')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.5)
        
        st.pyplot(fig)


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">🌊 SynthWave Deepfake Detector</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Deepfake Detection with Visual Explainability</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_options = {
        'Custom CNN': 'outputs/best_model_custom.keras',
        'MobileNetV2': 'outputs/best_model_mobilenet.keras',
        'EfficientNetB0': 'outputs/best_model_efficientnet.keras'
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    model_path = model_options[selected_model]
    
    # Load model
    with st.spinner('Loading model...'):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please ensure the model file exists.")
        st.info("Train the model first by running: `python 03_train_model.py`")
        return
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    st.sidebar.success(f"✓ {selected_model} loaded successfully!")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["📷 Image Upload", "🎥 Video Upload", "📹 Webcam (Coming Soon)"]
    )
    
    # Main content
    if mode == "📷 Image Upload":
        st.markdown("## Upload an Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a face image to check if it's real or fake"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert to RGB if needed
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Predict
            with st.spinner('Analyzing image...'):
                class_idx, confidence, heatmap, superimposed, img_resized = predict_image(
                    model, image_np, gradcam
                )
            
            # Display results
            display_results(class_idx, confidence, img_resized, heatmap, superimposed)
    
    elif mode == "🎥 Video Upload":
        st.markdown("## Upload a Video")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video to analyze frame by frame"
        )
        
        max_frames = st.slider("Max frames to analyze", 10, 100, 30)
        
        if uploaded_video is not None:
            with st.spinner('Processing video...'):
                process_video(model, gradcam, uploaded_video, max_frames)
    
    else:  # Webcam mode
        st.info("🚧 Webcam mode coming soon! This will allow real-time detection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using TensorFlow and Streamlit</p>
            <p>SynthWave Deepfake Detector v1.0</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()