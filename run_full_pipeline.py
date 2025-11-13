"""
SynthWave Master Pipeline Script
Automated execution of the complete deepfake detection pipeline
Run this after preparing your dataset with 01_dataset_preparation.py

Usage:
    python run_full_pipeline.py --model-type custom --epochs 50
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}{Colors.ENDC}\n")


def print_step(step_num, total_steps, title):
    """Print step information"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}/{total_steps}] {title}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'─'*70}{Colors.ENDC}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def check_prerequisites():
    """Check if dataset and dependencies are ready"""
    print_step(0, 7, "Checking Prerequisites")
    
    all_good = True
    
    # Check if dataset is prepared
    data_dir = Path('data/processed')
    required_dirs = [
        'train/real', 'train/fake',
        'val/real', 'val/fake',
        'test/real', 'test/fake'
    ]
    
    print("\n📁 Checking dataset structure...")
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if full_path.exists():
            image_count = len(list(full_path.glob('*.jpg')) + list(full_path.glob('*.png')))
            if image_count > 0:
                print_success(f"{dir_path}: {image_count} images")
            else:
                print_error(f"{dir_path}: No images found!")
                all_good = False
        else:
            print_error(f"{dir_path}: Directory not found!")
            all_good = False
    
    # Check required Python modules
    print("\n📦 Checking required modules...")
    required_modules = [
        'tensorflow', 'keras', 'numpy', 'pandas', 'sklearn',
        'cv2', 'PIL', 'mtcnn', 'matplotlib', 'seaborn', 'streamlit'
    ]
    
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                from PIL import Image
            else:
                __import__(module)
            print_success(f"{module}")
        except ImportError:
            print_error(f"{module} not installed!")
            all_good = False
    
    if not all_good:
        print_error("\n❌ Prerequisites check failed!")
        print_warning("Please run: python 01_dataset_preparation.py first")
        print_warning("And ensure all dependencies are installed: pip install -r requirements.txt")
        return False
    
    print_success("\n✅ All prerequisites satisfied!")
    return True


def run_model_architecture_demo():
    """Run model architecture demonstration"""
    print_step(1, 7, "Model Architecture Overview")
    
    try:
        # Import and show model architectures
        from model_architecture import DeepfakeDetector, print_model_summary
        
        print("\n🏗️ Available model architectures:")
        print("  1. Custom CNN (Lightweight, ~2.5M params)")
        print("  2. MobileNetV2 (Transfer Learning, ~3.5M params)")
        print("  3. EfficientNetB0 (Best Accuracy, ~5M params)")
        
        print_success("Model architectures loaded successfully")
        return True
        
    except Exception as e:
        print_error(f"Error loading model architectures: {str(e)}")
        return False


def run_training(model_type='custom', epochs=50, batch_size=32):
    """Run model training"""
    print_step(2, 7, f"Training {model_type.upper()} Model")
    
    start_time = time.time()
    
    try:
        print(f"\n🚀 Starting training with:")
        print(f"   • Model: {model_type}")
        print(f"   • Epochs: {epochs}")
        print(f"   • Batch size: {batch_size}")
        print(f"\n⏳ This may take 20-40 minutes (CPU) or 5-10 minutes (GPU)...\n")
        
        # Import training module
        from train_model import DeepfakeTrainer
        
        # Initialize trainer
        trainer = DeepfakeTrainer(
            data_dir='data/processed',
            model_type=model_type,
            img_size=(224, 224),
            batch_size=batch_size
        )
        
        # Create data generators
        trainer.create_data_generators()
        
        # Build model
        trainer.build_model()
        
        # Train model
        trainer.train(epochs=epochs, patience=10)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        metrics = trainer.evaluate_model()
        
        elapsed_time = time.time() - start_time
        
        print_success(f"\n✅ Training completed in {elapsed_time/60:.1f} minutes!")
        print(f"\n📊 Results:")
        print(f"   • Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
        print(f"   • Test Precision: {metrics['test_precision']*100:.2f}%")
        print(f"   • Test Recall: {metrics['test_recall']*100:.2f}%")
        print(f"   • ROC AUC: {metrics['roc_auc']:.4f}")
        
        return True, metrics
        
    except Exception as e:
        print_error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def generate_gradcam_explanations(model_type='custom', num_samples=10):
    """Generate Grad-CAM visualizations"""
    print_step(3, 7, "Generating Grad-CAM Explanations")
    
    try:
        from tensorflow import keras
        from gradcam_explainability import GradCAM, batch_explain
        
        model_path = f'outputs/best_model_{model_type}.keras'
        
        print(f"\n🔍 Loading model: {model_path}")
        model = keras.models.load_model(model_path)
        print_success("Model loaded successfully")
        
        print(f"\n📸 Generating {num_samples} Grad-CAM visualizations...")
        
        # Generate explanations
        batch_explain(
            model=model,
            image_dir='data/processed/test',
            output_dir='outputs/gradcam',
            num_samples=num_samples,
            class_names=['Fake', 'Real']
        )
        
        print_success(f"✅ Grad-CAM visualizations saved to outputs/gradcam/")
        return True
        
    except Exception as e:
        print_error(f"Grad-CAM generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline_tests(model_type='custom'):
    """Run comprehensive pipeline tests"""
    print_step(4, 7, "Running Pipeline Tests")
    
    try:
        from test_pipeline import PipelineTester
        
        model_path = f'outputs/best_model_{model_type}.keras'
        
        print(f"\n🧪 Testing model: {model_path}\n")
        
        tester = PipelineTester(
            model_path=model_path,
            test_dir='data/processed/test'
        )
        
        # Load model
        if not tester.load_model():
            print_error("Failed to load model")
            return False
        
        # Run individual tests
        print("\n1️⃣ Testing batch predictions...")
        batch_results = tester.test_batch_predictions(num_samples=5)
        
        print("\n2️⃣ Testing Grad-CAM generation...")
        gradcam_success = tester.test_gradcam_generation(num_samples=3)
        
        print("\n3️⃣ Testing model performance...")
        performance_metrics = tester.test_model_performance()
        
        print("\n4️⃣ Testing inference speed...")
        speed_metrics = tester.test_inference_speed(num_iterations=100)
        
        # Generate report
        report = tester.generate_test_report(
            batch_results, 
            performance_metrics, 
            speed_metrics
        )
        
        print_success("\n✅ All tests completed successfully!")
        return True, report
        
    except Exception as e:
        print_error(f"Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def analyze_sample_video(model_type='custom'):
    """Analyze a sample video if available"""
    print_step(5, 7, "Video Analysis (Optional)")
    
    # Check if any video files exist
    video_extensions = ['.mp4', '.avi', '.mov']
    sample_videos = []
    
    for ext in video_extensions:
        sample_videos.extend(Path('.').rglob(f'*{ext}'))
    
    if not sample_videos:
        print_warning("No sample videos found. Skipping video analysis.")
        print("   To test video analysis, place a video file and run:")
        print(f"   python 06_video_processor.py --video path/to/video.mp4 --model outputs/best_model_{model_type}.keras")
        return True
    
    print(f"\n🎥 Found {len(sample_videos)} video(s). Analyzing first video...")
    
    try:
        from video_processor import VideoDeepfakeDetector
        
        model_path = f'outputs/best_model_{model_type}.keras'
        video_path = sample_videos[0]
        
        print(f"   Video: {video_path.name}")
        
        detector = VideoDeepfakeDetector(
            model_path=model_path,
            sampling_rate=30
        )
        
        results = detector.analyze_video(
            video_path=video_path,
            output_dir='outputs/video_analysis'
        )
        
        print_success("\n✅ Video analysis completed!")
        print(f"   • Verdict: {results['verdict']}")
        print(f"   • Confidence: {results['verdict_confidence']:.2f}%")
        print(f"   • Frames analyzed: {results['frames_analyzed']}")
        
        return True
        
    except Exception as e:
        print_error(f"Video analysis failed: {str(e)}")
        print_warning("This is optional. Continuing...")
        return True


def prepare_streamlit_launch(model_type='custom'):
    """Prepare information for Streamlit launch"""
    print_step(6, 7, "Preparing Web Interface")
    
    print("\n🌐 Streamlit app is ready to launch!")
    print("\n📝 To start the web interface, run:")
    print(f"\n   {Colors.OKGREEN}streamlit run 05_streamlit_app.py{Colors.ENDC}")
    print("\n✨ Features available:")
    print("   • Upload and analyze images")
    print("   • Upload and analyze videos")
    print("   • View Grad-CAM explanations")
    print("   • Switch between model architectures")
    print("   • Real-time confidence scores")
    
    print_success("\nWeb interface is ready!")
    return True


def generate_final_report(training_metrics, test_report, model_type, total_time):
    """Generate comprehensive final report"""
    print_step(7, 7, "Generating Final Report")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'total_execution_time_minutes': total_time / 60,
        'training_metrics': training_metrics,
        'test_report': test_report,
        'files_generated': {
            'model': f'outputs/best_model_{model_type}.keras',
            'training_plots': f'outputs/training_history_{model_type}.png',
            'confusion_matrix': f'outputs/confusion_matrix_{model_type}.png',
            'roc_curve': f'outputs/roc_curve_{model_type}.png',
            'gradcam_visualizations': 'outputs/gradcam/',
            'test_report': 'outputs/test_report.json',
            'final_report': 'outputs/pipeline_report.json'
        }
    }
    
    # Save report
    report_path = Path('outputs/pipeline_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print summary
    print("\n" + "="*70)
    print("🎉 SYNTHWAVE PIPELINE EXECUTION COMPLETE!")
    print("="*70)
    
    print(f"\n⏱️  Total Execution Time: {total_time/60:.1f} minutes")
    print(f"\n🎯 Model Performance:")
    if training_metrics:
        print(f"   • Accuracy: {training_metrics['test_accuracy']*100:.2f}%")
        print(f"   • Precision: {training_metrics['test_precision']*100:.2f}%")
        print(f"   • Recall: {training_metrics['test_recall']*100:.2f}%")
        print(f"   • ROC AUC: {training_metrics['roc_auc']:.4f}")
    
    if test_report and 'speed_metrics' in test_report:
        print(f"\n⚡ Inference Speed:")
        print(f"   • Average: {test_report['speed_metrics']['avg_time_ms']:.2f} ms")
        print(f"   • FPS: {test_report['speed_metrics']['fps']:.2f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • Model: outputs/best_model_{model_type}.keras")
    print(f"   • Training plots: outputs/training_history_{model_type}.png")
    print(f"   • Evaluation plots: outputs/confusion_matrix_{model_type}.png")
    print(f"   • Grad-CAM examples: outputs/gradcam/")
    print(f"   • Test report: outputs/test_report.json")
    print(f"   • Final report: outputs/pipeline_report.json")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review the generated visualizations in outputs/")
    print(f"   2. Launch web interface: streamlit run 05_streamlit_app.py")
    print(f"   3. Test on your own images")
    print(f"   4. Prepare your hackathon presentation!")
    
    print("\n" + "="*70)
    
    print_success(f"✅ Full report saved to {report_path}")
    
    return True


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='SynthWave Full Pipeline Execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (Custom CNN, 50 epochs)
  python run_full_pipeline.py
  
  # Run with MobileNetV2 for 30 epochs
  python run_full_pipeline.py --model-type mobilenet --epochs 30
  
  # Quick run with fewer epochs for testing
  python run_full_pipeline.py --epochs 20 --batch-size 16
  
  # Skip video analysis
  python run_full_pipeline.py --skip-video
        """
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='custom',
        choices=['custom', 'mobilenet', 'efficientnet'],
        help='Model architecture to use (default: custom)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--gradcam-samples',
        type=int,
        default=10,
        help='Number of Grad-CAM samples to generate (default: 10)'
    )
    
    parser.add_argument(
        '--skip-video',
        action='store_true',
        help='Skip video analysis step'
    )
    
    parser.add_argument(
        '--auto-launch',
        action='store_true',
        help='Automatically launch Streamlit app after completion'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_header("🌊 SYNTHWAVE FULL PIPELINE EXECUTION 🌊")
    
    print(f"\n{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"   • Model Type: {args.model_type}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Batch Size: {args.batch_size}")
    print(f"   • Grad-CAM Samples: {args.gradcam_samples}")
    print(f"   • Skip Video: {args.skip_video}")
    
    start_time = time.time()
    
    # Step 0: Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Model architecture demo
    if not run_model_architecture_demo():
        print_error("Failed at model architecture step")
        sys.exit(1)
    
    # Step 2: Train model
    training_success, training_metrics = run_training(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if not training_success:
        print_error("Training failed. Exiting pipeline.")
        sys.exit(1)
    
    # Step 3: Generate Grad-CAM explanations
    if not generate_gradcam_explanations(
        model_type=args.model_type,
        num_samples=args.gradcam_samples
    ):
        print_warning("Grad-CAM generation failed, but continuing...")
    
    # Step 4: Run tests
    test_success, test_report = run_pipeline_tests(model_type=args.model_type)
    
    if not test_success:
        print_warning("Testing failed, but continuing...")
    
    # Step 5: Video analysis (optional)
    if not args.skip_video:
        analyze_sample_video(model_type=args.model_type)
    else:
        print_warning("Skipping video analysis as requested")
    
    # Step 6: Prepare Streamlit
    prepare_streamlit_launch(model_type=args.model_type)
    
    # Step 7: Generate final report
    total_time = time.time() - start_time
    generate_final_report(
        training_metrics=training_metrics,
        test_report=test_report,
        model_type=args.model_type,
        total_time=total_time
    )
    
    # Auto-launch Streamlit if requested
    if args.auto_launch:
        print(f"\n{Colors.OKGREEN}🚀 Launching Streamlit app...{Colors.ENDC}")
        try:
            subprocess.run(['streamlit', 'run', '05_streamlit_app.py'])
        except Exception as e:
            print_error(f"Failed to launch Streamlit: {str(e)}")
            print("Please run manually: streamlit run 05_streamlit_app.py")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✅ Pipeline execution completed successfully!{Colors.ENDC}")
    print(f"\n{Colors.OKCYAN}Happy hacking! 🌊{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠️  Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)