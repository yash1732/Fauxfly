# 🌊 SynthWave: Lightweight Deepfake Detection System

**An end-to-end deepfake detection system with visual explainability using Grad-CAM**

Built for hackathons, educational purposes, and rapid prototyping of deepfake detection solutions.

---

## 📋 Features

- ✅ **Image-based Deepfake Detection** - Detect manipulated faces in images
- ✅ **Video Analysis** - Frame-by-frame video analysis with aggregated predictions
- ✅ **Multiple Model Architectures** - Custom CNN, MobileNetV2, and EfficientNetB0
- ✅ **Grad-CAM Explainability** - Visual explanation of model decisions
- ✅ **Interactive Web Interface** - Streamlit app for easy testing
- ✅ **Comprehensive Metrics** - Accuracy, precision, recall, F1-score, ROC curves
- ✅ **CPU-Friendly** - Lightweight models that run on CPU

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository (or create new directory)
mkdir synthwave && cd synthwave

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Create directory structure
python download_sample_dataset.py

# Place your images:
# - Real images in: data/raw/real/
# - Fake images in: data/raw/fake/

# Process dataset (face detection, cropping, splitting)
python 01_dataset_preparation.py
```

**Dataset Recommendations:**
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **CelebDF**: http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html
- **140K Real and Fake Faces**: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

### 3. Train Model

```bash
# Train custom CNN (fastest, good for hackathons)
python 03_train_model.py

# Or modify MODEL_TYPE in the script to use:
# - 'mobilenet' for better accuracy
# - 'efficientnet' for best accuracy (slightly slower)
```

### 4. Generate Explainability

```bash
# Generate Grad-CAM visualizations
python 04_gradcam_explainability.py
```

### 5. Launch Web Interface

```bash
# Start Streamlit app
streamlit run 05_streamlit_app.py
```

### 6. Analyze Videos (Optional)

```bash
# Analyze a video file
python 06_video_processor.py --video path/to/video.mp4 --output outputs/video_analysis
```

---

## 📁 Project Structure

```
synthwave/
├── requirements.txt                    # Dependencies
├── download_sample_dataset.py          # Dataset download guide
├── 01_dataset_preparation.py           # Face detection & preprocessing
├── 02_model_architecture.py            # CNN model definitions
├── 03_train_model.py                   # Training pipeline
├── 04_gradcam_explainability.py        # Grad-CAM implementation
├── 05_streamlit_app.py                 # Web interface
├── 06_video_processor.py               # Video analysis module
├── data/
│   ├── raw/
│   │   ├── real/                       # Raw real images
│   │   └── fake/                       # Raw fake images
│   └── processed/
│       ├── train/
│       │   ├── real/                   # Processed training images
│       │   └── fake/
│       ├── val/
│       │   ├── real/                   # Processed validation images
│       │   └── fake/
│       └── test/
│           ├── real/                   # Processed test images
│           └── fake/
└── outputs/
    ├── best_model_custom.keras         # Trained models
    ├── training_history_custom.png     # Training plots
    ├── confusion_matrix_custom.png     # Evaluation plots
    └── gradcam/                        # Grad-CAM visualizations
```

---

## 🏗️ Model Architectures

### 1. Custom CNN (Recommended for Hackathons)
- **Parameters**: ~2.5M
- **Inference Speed**: ~50ms per image (CPU)
- **Architecture**: 4 Conv blocks + Dense layers
- **Best for**: Quick prototyping, CPU deployment

### 2. MobileNetV2 (Transfer Learning)
- **Parameters**: ~3.5M (only top layers trainable)
- **Inference Speed**: ~100ms per image (CPU)
- **Architecture**: Pre-trained MobileNetV2 + custom head
- **Best for**: Better accuracy with reasonable speed

### 3. EfficientNetB0 (Transfer Learning)
- **Parameters**: ~5M (only top layers trainable)
- **Inference Speed**: ~150ms per image (CPU)
- **Architecture**: Pre-trained EfficientNetB0 + custom head
- **Best for**: Best accuracy when speed is not critical

---

## 📊 Training Details

### Data Augmentation
- Random rotation (±20°)
- Horizontal flipping
- Width/height shifts (20%)
- Zoom (20%)
- Brightness adjustments

### Training Configuration
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Callbacks**: 
  - Early stopping (patience=10)
  - Model checkpoint
  - Learning rate reduction
  - TensorBoard logging

### Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 🎯 Usage Examples

### Example 1: Single Image Prediction

```python
from tensorflow import keras
import cv2
import numpy as np

# Load model
model = keras.models.load_model('outputs/best_model_custom.keras')

# Load and preprocess image
img = cv2.imread('test_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)) / 255.0
img_batch = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img_batch)
class_idx = np.argmax(prediction[0])
confidence = prediction[0][class_idx] * 100

print(f"Prediction: {'Fake' if class_idx == 0 else 'Real'}")
print(f"Confidence: {confidence:.2f}%")
```

### Example 2: Grad-CAM Visualization

```python
from gradcam_explainability import GradCAM

# Initialize Grad-CAM
gradcam = GradCAM(model)

# Generate explanation
gradcam.explain_prediction(
    image_path='test_image.jpg',
    model=model,
    class_names=['Fake', 'Real'],
    save_path='gradcam_result.png'
)
```

### Example 3: Video Analysis

```python
from video_processor import VideoDeepfakeDetector

# Initialize detector
detector = VideoDeepfakeDetector(
    model_path='outputs/best_model_custom.keras',
    sampling_rate=30
)

# Analyze video
results = detector.analyze_video(
    video_path='test_video.mp4',
    output_dir='outputs/video_analysis'
)

print(f"Verdict: {results['verdict']}")
print(f"Confidence: {results['verdict_confidence']:.2f}%")
```

---

## 🎨 Streamlit Interface Features

1. **Image Upload Mode**
   - Upload face images
   - Get instant predictions
   - View Grad-CAM heatmaps
   - See confidence scores

2. **Video Upload Mode**
   - Upload video files
   - Frame-by-frame analysis
   - Temporal analysis plots
   - Overall verdict

3. **Model Selection**
   - Switch between trained models
   - Compare performance

---

## 📈 Expected Results

### Training Performance (on sample dataset)
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 75-85%

*Note: Results vary based on dataset quality and size*

### Inference Speed (CPU)
- **Custom CNN**: ~50ms per image
- **MobileNetV2**: ~100ms per image
- **EfficientNetB0**: ~150ms per image

---

## 🐛 Troubleshooting

### Issue: "No faces detected"
**Solution**: Ensure images contain clear, frontal faces. Adjust MTCNN detection parameters if needed.

### Issue: Low accuracy
**Solutions**:
- Increase dataset size
- Add more augmentation
- Train for more epochs
- Try different model architectures

### Issue: Out of memory
**Solutions**:
- Reduce batch size
- Use custom CNN instead of transfer learning
- Process fewer frames for videos

### Issue: Model not found
**Solution**: Ensure you've run training script first: `python 03_train_model.py`

---

## 🎓 For Hackathons

### Quick Demo Setup (1 hour)
1. Use pre-collected small dataset (100-200 images per class)
2. Train custom CNN for 20-30 epochs
3. Generate 5-10 Grad-CAM examples
4. Launch Streamlit app for live demo

### Presentation Tips
- Show live predictions on judges' photos
- Demonstrate Grad-CAM explainability
- Compare real vs fake detection
- Discuss potential applications

### Scoring Points
- ✅ Working end-to-end system
- ✅ Visual explainability
- ✅ Interactive demo
- ✅ Video analysis capability
- ✅ Comprehensive evaluation metrics

---

## 🔮 Future Enhancements

- [ ] Real-time webcam detection
- [ ] Audio deepfake detection
- [ ] Multi-face detection in single image
- [ ] Mobile app deployment
- [ ] Temporal consistency checking for videos
- [ ] Ensemble of multiple models
- [ ] API endpoint deployment

---

## 📚 References

1. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
2. **FaceForensics++**: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images"
3. **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
4. **EfficientNet**: Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

---

## 📝 License

MIT License - Feel free to use for hackathons, education, and research!

---

## 🤝 Contributing

Contributions welcome! Please open issues or pull requests.

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the development team.

---

**Built with ❤️ for the AI community**