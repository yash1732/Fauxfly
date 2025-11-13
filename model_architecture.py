"""
SynthWave Model Architectures
Contains custom CNN and transfer learning models
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

class DeepfakeDetector:
    """Factory class for creating deepfake detection models"""
    
    @staticmethod
    def create_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
        """
        Create a lightweight custom CNN architecture
        
        Architecture:
        - 4 Convolutional blocks with BatchNorm and MaxPooling
        - Global Average Pooling
        - Dense layers with Dropout
        - Binary classification output
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes (2 for binary)
            
        Returns:
            Compiled Keras model
        """
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Activation('relu', name='relu1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.Activation('relu', name='relu2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Activation('relu', name='relu3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            # Block 4
            layers.Conv2D(256, (3, 3), padding='same', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.Activation('relu', name='relu4'),
            layers.MaxPooling2D((2, 2), name='pool4'),
            layers.Dropout(0.25, name='dropout4'),
            
            # Classification head
            layers.GlobalAveragePooling2D(name='gap'),
            layers.Dense(512, activation='relu', name='fc1'),
            layers.Dropout(0.5, name='dropout5'),
            layers.Dense(128, activation='relu', name='fc2'),
            layers.Dropout(0.5, name='dropout6'),
            layers.Dense(num_classes, activation='softmax', name='output')
        ], name='SynthWave_CustomCNN')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    @staticmethod
    def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=2, trainable_layers=20):
        """
        Create transfer learning model using MobileNetV2
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            trainable_layers: Number of top layers to fine-tune
            
        Returns:
            Compiled Keras model
        """
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base layers
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        
        # Build complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(name='gap'),
            layers.Dense(256, activation='relu', name='fc1'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(128, activation='relu', name='fc2'),
            layers.Dropout(0.5, name='dropout2'),
            layers.Dense(num_classes, activation='softmax', name='output')
        ], name='SynthWave_MobileNetV2')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    @staticmethod
    def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=2, trainable_layers=30):
        """
        Create transfer learning model using EfficientNetB0
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            trainable_layers: Number of top layers to fine-tune
            
        Returns:
            Compiled Keras model
        """
        
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base layers
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        
        # Build complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(name='gap'),
            layers.Dense(256, activation='relu', name='fc1'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(128, activation='relu', name='fc2'),
            layers.Dropout(0.5, name='dropout2'),
            layers.Dense(num_classes, activation='softmax', name='output')
        ], name='SynthWave_EfficientNetB0')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model


def print_model_summary(model):
    """Print detailed model summary with parameter count"""
    print("\n" + "="*70)
    print(f"MODEL: {model.name}")
    print("="*70)
    model.summary()
    
    # Calculate parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print("\n" + "="*70)
    print("PARAMETER COUNT")
    print("="*70)
    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("\n🚀 SynthWave Model Architecture Demo\n")
    
    # Create custom CNN
    print("1. Creating Custom CNN...")
    custom_model = DeepfakeDetector.create_custom_cnn()
    print_model_summary(custom_model)
    
    # Create MobileNetV2 model
    print("\n2. Creating MobileNetV2 Transfer Learning Model...")
    mobilenet_model = DeepfakeDetector.create_mobilenet_model()
    print_model_summary(mobilenet_model)
    
    # Create EfficientNet model
    print("\n3. Creating EfficientNetB0 Transfer Learning Model...")
    efficientnet_model = DeepfakeDetector.create_efficientnet_model()
    print_model_summary(efficientnet_model)
    
    print("✓ All models created successfully!")
    print("\nRecommendation: For hackathon, use Custom CNN (fastest) or MobileNetV2 (best accuracy/speed trade-off)")