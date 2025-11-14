"""
SynthWave Video Processing Module
Handles video-level deepfake detection by aggregating frame predictions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from tensorflow import keras

# ✅ FIXED: Import from the correct file name 'gradcam_explainability.py'
from gradcam_explainability import GradCAM


class VideoDeepfakeDetector:
    """Video-level deepfake detection"""
    
    def __init__(self, model_path, sampling_rate=30):
        """
        Initialize video detector
        
        Args:
            model_path: Path to trained model
            sampling_rate: Sample every Nth frame (default: 30, i.e., 1 FPS for 30 FPS video)
        """
        self.model = keras.models.load_model(model_path)
        self.gradcam = GradCAM(self.model)
        self.sampling_rate = sampling_rate
        self.class_names = ['Fake', 'Real']
    
    def extract_frames(self, video_path, max_frames=None):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            frames: List of frames
            metadata: Video metadata
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': total_frames / fps if fps > 0 else 0
        }
        
        # Determine frames to sample
        if max_frames:
            frame_indices = np.linspace(0, total_frames - 1, 
                                       min(max_frames, total_frames), 
                                       dtype=int)
        else:
            frame_indices = range(0, total_frames, self.sampling_rate)
        
        frames = []
        for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    'index': frame_idx,
                    'timestamp': frame_idx / fps if fps > 0 else 0,
                    'image': frame_rgb
                })
        
        cap.release()
        
        print(f"✓ Extracted {len(frames)} frames from {total_frames} total frames")
        
        return frames, metadata
    
    def analyze_video(self, video_path, output_dir=None, generate_report=True):
        """
        Analyze entire video for deepfakes
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            generate_report: Whether to generate detailed report
            
        Returns:
            results: Dictionary containing analysis results
        """
        video_path = Path(video_path)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎥 Analyzing video: {video_path.name}")
        
        # Extract frames
        frames, metadata = self.extract_frames(video_path)
        
        # Analyze each frame
        predictions = []
        
        for frame_data in tqdm(frames, desc="Analyzing frames"):
            img = frame_data['image']
            
            # Preprocess
            img_resized = cv2.resize(img, (224, 224))
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            pred = self.model.predict(img_batch, verbose=0)
            class_idx = np.argmax(pred[0])
            confidence = pred[0][class_idx]
            
            predictions.append({
                'frame_index': frame_data['index'],
                'timestamp': frame_data['timestamp'],
                'class': class_idx,
                'class_name': self.class_names[class_idx],
                'confidence': float(confidence),
                'probabilities': pred[0].tolist()
            })
        
        # Aggregate results
        fake_count = sum(1 for p in predictions if p['class'] == 0)
        real_count = len(predictions) - fake_count
        fake_percentage = (fake_count / len(predictions)) * 100
        
        avg_fake_confidence = np.mean([p['confidence'] for p in predictions if p['class'] == 0]) if fake_count > 0 else 0
        avg_real_confidence = np.mean([p['confidence'] for p in predictions if p['class'] == 1]) if real_count > 0 else 0
        
        # Determine overall verdict
        if fake_percentage > 50:
            verdict = "FAKE"
            verdict_confidence = fake_percentage
        else:
            verdict = "REAL"
            verdict_confidence = 100 - fake_percentage
        
        results = {
            'video_path': str(video_path),
            'metadata': metadata,
            'frames_analyzed': len(predictions),
            'fake_frames': fake_count,
            'real_frames': real_count,
            'fake_percentage': fake_percentage,
            'verdict': verdict,
            'verdict_confidence': verdict_confidence,
            'avg_fake_confidence': float(avg_fake_confidence),
            'avg_real_confidence': float(avg_real_confidence),
            'frame_predictions': predictions
        }
        
        # Generate report
        if generate_report and output_dir:
            self.generate_report(results, output_dir)
        
        return results
    
    def generate_report(self, results, output_dir):
        """Generate visual report of analysis"""
        
        output_dir = Path(output_dir)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f"SynthWave Video Analysis Report\n{Path(results['video_path']).name}", 
                    fontsize=16, fontweight='bold')
        
        # 1. Summary statistics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        summary_text = f"""
        VIDEO METADATA:
        • Duration: {results['metadata']['duration']:.2f} seconds
        • Total Frames: {results['metadata']['total_frames']}
        • FPS: {results['metadata']['fps']}
        • Resolution: {results['metadata']['width']}x{results['metadata']['height']}
        
        ANALYSIS RESULTS:
        • Frames Analyzed: {results['frames_analyzed']}
        • Fake Frames: {results['fake_frames']} ({results['fake_percentage']:.1f}%)
        • Real Frames: {results['real_frames']} ({100-results['fake_percentage']:.1f}%)
        • Overall Verdict: {results['verdict']} (Confidence: {results['verdict_confidence']:.1f}%)
        """
        
        ax1.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 2. Pie chart - Fake vs Real distribution
        ax2 = fig.add_subplot(gs[1, 0])
        sizes = [results['fake_frames'], results['real_frames']]
        colors = ['#ff6b6b', '#51cf66']
        labels = ['Fake', 'Real']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Frame Classification Distribution', fontweight='bold')
        
        # 3. Timeline plot
        ax3 = fig.add_subplot(gs[1, 1])
        timestamps = [p['timestamp'] for p in results['frame_predictions']]
        classes = [p['class'] for p in results['frame_predictions']]
        colors_scatter = ['red' if c == 0 else 'green' for c in classes]
        
        ax3.scatter(timestamps, classes, c=colors_scatter, alpha=0.6, s=50)
        ax3.set_xlabel('Time (seconds)', fontweight='bold')
        ax3.set_ylabel('Prediction', fontweight='bold')
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Fake', 'Real'])
        ax3.set_title('Frame-by-Frame Timeline', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Confidence distribution
        ax4 = fig.add_subplot(gs[2, 0])
        fake_confidences = [p['confidence'] for p in results['frame_predictions'] if p['class'] == 0]
        real_confidences = [p['confidence'] for p in results['frame_predictions'] if p['class'] == 1]
        
        if fake_confidences:
            ax4.hist(fake_confidences, bins=20, alpha=0.6, color='red', label='Fake', edgecolor='black')
        if real_confidences:
            ax4.hist(real_confidences, bins=20, alpha=0.6, color='green', label='Real', edgecolor='black')
        
        ax4.set_xlabel('Confidence Score', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Confidence Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Moving average plot
        ax5 = fig.add_subplot(gs[2, 1])
        window_size = min(10, len(results['frame_predictions']))
        
        fake_probs = [p['probabilities'][0] for p in results['frame_predictions']]
        moving_avg = np.convolve(fake_probs, np.ones(window_size)/window_size, mode='valid')
        
        ax5.plot(moving_avg, linewidth=2, color='purple')
        ax5.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Decision Threshold')
        ax5.fill_between(range(len(moving_avg)), moving_avg, 0.5, 
                         where=(np.array(moving_avg) > 0.5), alpha=0.3, color='red', label='Fake Region')
        ax5.fill_between(range(len(moving_avg)), moving_avg, 0.5, 
                         where=(np.array(moving_avg) <= 0.5), alpha=0.3, color='green', label='Real Region')
        ax5.set_xlabel('Frame', fontweight='bold')
        ax5.set_ylabel('Fake Probability (Moving Avg)', fontweight='bold')
        ax5.set_title(f'Temporal Analysis (Window={window_size})', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Save figure
        report_path = output_dir / 'video_analysis_report.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Report saved to {report_path}")
        
        # Save JSON results
        json_path = output_dir / 'video_analysis_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Results saved to {json_path}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SynthWave Video Deepfake Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='outputs/best_model_custom.keras',
                       help='Path to trained model')
    parser.add_I('output', type=str, default='outputs/video_analysis',
                       help='Output directory for results')
    parser.add_argument('--sampling-rate', type=int, default=30,
                       help='Sample every Nth frame')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🌊 SYNTHWAVE VIDEO DEEPFAKE DETECTOR")
    print("="*70)
    
    # Initialize detector
    detector = VideoDeepfakeDetector(
        model_path=args.model,
        sampling_rate=args.sampling_rate
    )
    
    # Analyze video
    results = detector.analyze_video(
        video_path=args.video,
        output_dir=args.output
    )
    
    # Print summary
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    print(f"Verdict: {results['verdict']}")
    print(f"Confidence: {results['verdict_confidence']:.2f}%")
    print(f"Fake Frames: {results['fake_frames']}/{results['frames_analyzed']} ({results['fake_percentage']:.1f}%)")
    print("="*70)