"""
SynthWave Pipeline Testing Script
Comprehensive testing of the entire deepfake detection pipeline
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tensorflow import keras

# ✅ IMPORT: Import the new functions
from gradcam_explainability import (
    make_gradcam_heatmap, 
    explain_prediction,
    find_last_conv_layer_name
)


class PipelineTester:
    """Test the complete SynthWave pipeline"""
    
    def __init__(self, model_path, test_dir='data/processed/test'):
        """
        Initialize pipeline tester
        
        Args:
            model_path: Path to trained model
            test_dir: Directory containing test images
        """
        self.model_path = Path(model_path)
        self.test_dir = Path(test_dir)
        self.model = None
        self.last_conv_layer_name = None # ✅ ADD: Store layer name
        self.class_names = ['Fake', 'Real']
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            # Build the model on load
            if not self.model.built:
                self.model(tf.zeros((1, 224, 224, 3)))
            
            # ✅ EXECUTE: Find and store the layer name
            self.last_conv_layer_name = find_last_conv_layer_name(self.model)
            
            print(f"✓ Model loaded: {self.model_path.name}")
            print(f"✓ Using Grad-CAM layer: '{self.last_conv_layer_name}'")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    
    def test_single_prediction(self, image_path):
        """Test prediction on a single image"""
        print(f"\n📸 Testing: {Path(image_path).name}")
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            predictions = self.model(img_batch, training=False) # Use __call__
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx] * 100
            
            # Generate Grad-CAM
            # ✅ EXECUTE: Use the new function
            heatmap = make_gradcam_heatmap(
                img_batch, self.model, self.last_conv_layer_name, class_idx
            )
            
            result = {
                'image': image_path,
                'prediction': self.class_names[class_idx],
                'confidence': confidence,
                'probabilities': {
                    'fake': float(predictions[0][0] * 100),
                    'real': float(predictions[0][1] * 100)
                },
                'success': True
            }
            
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  Fake prob: {result['probabilities']['fake']:.2f}%")
            print(f"  Real prob: {result['probabilities']['real']:.2f}%")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def test_batch_predictions(self, num_samples=5):
        """Test predictions on batch of images"""
        print(f"\n📦 Testing batch predictions ({num_samples} samples per class)...")
        
        results = {'fake': [], 'real': []}
        
        for class_name in ['fake', 'real']:
            class_dir = self.test_dir / class_name
            
            if not class_dir.exists():
                print(f"  ✗ Directory not found: {class_dir}")
                continue
            
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if len(image_files) == 0:
                print(f"  ✗ No images found in {class_dir}")
                continue
            
            # Sample images
            sampled = np.random.choice(image_files, 
                                      min(num_samples, len(image_files)), 
                                      replace=False)
            
            for img_path in sampled:
                result = self.test_single_prediction(img_path)
                results[class_name].append(result)
        
        return results
    
    def test_gradcam_generation(self, num_samples=3):
        """Test Grad-CAM generation"""
        print(f"\n🔍 Testing Grad-CAM generation ({num_samples} samples)...")
        
        output_dir = Path('outputs/test_gradcam')
        output_dir.mkdir(exist_ok=True)
        
        # Get random test images
        all_images = list(self.test_dir.rglob('*.jpg')) + list(self.test_dir.rglob('*.png'))
        sampled = np.random.choice(all_images, min(num_samples, len(all_images)), replace=False)
        
        success_count = 0
        
        for i, img_path in enumerate(sampled):
            try:
                save_path = output_dir / f'test_gradcam_{i+1}.png'
                # ✅ EXECUTE: Use the new function
                explain_prediction(
                    model=self.model,
                    last_conv_layer_name=self.last_conv_layer_name,
                    image_path=img_path,
                    class_names=self.class_names,
                    save_path=save_path
                )
                success_count += 1
                print(f"  ✓ Generated: {save_path.name}")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        print(f"\n  Success rate: {success_count}/{len(sampled)}")
        return success_count == len(sampled)
    
    def test_model_performance(self):
        """Test overall model performance metrics"""
        print("\n📊 Testing model performance...")
        
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            # Create data generator
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )
            
            # Evaluate
            results = self.model.evaluate(test_generator, verbose=0)
            
            metrics = {
                'loss': float(results[0]),
                'accuracy': float(results[1]),
                'precision': float(results[2]),
                'recall': float(results[3])
            }
            
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return None
    
    def test_inference_speed(self, num_iterations=100):
        """Test inference speed"""
        print(f"\n⚡ Testing inference speed ({num_iterations} iterations)...")
        
        import time
        
        # Create dummy input
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Warm up
        _ = self.model(dummy_input, training=False)
        
        # Measure speed
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.model(dummy_input, training=False)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        
        print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  FPS: {1000/avg_time:.2f}")
        
        return {'avg_time_ms': avg_time, 'std_time_ms': std_time, 'fps': 1000/avg_time}
    
    # ... (generate_test_report is unchanged) ...
    def generate_test_report(self, batch_results, performance_metrics, speed_metrics):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("📋 SYNTHWAVE TEST REPORT")
        print("="*70)
        
        # Count successes
        total_tests = sum(len(results) for results in batch_results.values())
        successful_tests = sum(
            sum(1 for r in results if r.get('success', False)) 
            for results in batch_results.values()
        )
        
        # Calculate accuracies
        fake_correct = sum(1 for r in batch_results['fake'] 
                          if r.get('success') and r['prediction'] == 'Fake')
        real_correct = sum(1 for r in batch_results['real'] 
                          if r.get('success') and r['prediction'] == 'Real')
        
        fake_total = len(batch_results['fake'])
        real_total = len(batch_results['real'])
        
        report = {
            'model': str(self.model_path),
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0
            },
            'class_performance': {
                'fake': {
                    'correct': fake_correct,
                    'total': fake_total,
                    'accuracy': fake_correct / fake_total if fake_total > 0 else 0
                },
                'real': {
                    'correct': real_correct,
                    'total': real_total,
                    'accuracy': real_correct / real_total if real_total > 0 else 0
                }
            },
            'overall_metrics': performance_metrics,
            'speed_metrics': speed_metrics
        }
        
        # Print report
        print(f"\nModel: {report['model']}")
        print(f"\nTest Execution:")
        print(f"  Total tests: {report['test_summary']['total_tests']}")
        print(f"  Successful: {report['test_summary']['successful_tests']}")
        print(f"  Success rate: {report['test_summary']['success_rate']*100:.2f}%")
        
        print(f"\nClass Performance:")
        print(f"  Fake detection: {report['class_performance']['fake']['correct']}/{report['class_performance']['fake']['total']} ({report['class_performance']['fake']['accuracy']*100:.2f}%)")
        print(f"  Real detection: {report['class_performance']['real']['correct']}/{report['class_performance']['real']['total']} ({report['class_performance']['real']['accuracy']*100:.2f}%)")
        
        if performance_metrics:
            print(f"\nOverall Metrics:")
            print(f"  Accuracy: {performance_metrics['accuracy']*100:.2f}%")
            print(f"  Precision: {performance_metrics['precision']*100:.2f}%")
            print(f"  Recall: {performance_metrics['recall']*100:.2f}%")
        
        if speed_metrics:
            print(f"\nInference Speed:")
            print(f"  Average: {speed_metrics['avg_time_ms']:.2f} ms")
            print(f"  FPS: {speed_metrics['fps']:.2f}")
        
        print("="*70)
        
        # Save report
        report_path = Path('outputs/test_report.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n✓ Report saved to {report_path}")
        
        return report

    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print("🧪 RUNNING SYNTHWAVE TEST SUITE")
        print("="*70)
        
        # Load model
        if not self.load_model():
            print("\n✗ Cannot proceed without model. Please train the model first.")
            return
        
        # Test 1: Batch predictions
        batch_results = self.test_batch_predictions(num_samples=5)
        
        # Test 2: Grad-CAM generation
        gradcam_success = self.test_gradcam_generation(num_samples=3)
        
        # Test 3: Model performance
        performance_metrics = self.test_model_performance()
        
        # Test 4: Inference speed
        speed_metrics = self.test_inference_speed(num_iterations=100)
        
        # Generate report
        report = self.generate_test_report(batch_results, performance_metrics, speed_metrics)
        
        print("\n✅ All tests completed!")


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SynthWave Pipeline Testing')
    parser.add_argument('--model', type=str, default='outputs/best_model_custom.keras',
                       help='Path to trained model')
    parser.add_argument('--test-dir', type=str, default='data/processed/test',
                       help='Test data directory')
    
    args = parser.parse_args()
    
    # Run tests
    tester = PipelineTester(model_path=args.model, test_dir=args.test_dir)
    tester.run_all_tests()


if __name__ == "__main__":
    main()