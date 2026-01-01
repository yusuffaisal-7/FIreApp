#!/usr/bin/env python3
"""
FIRE DETECTION FIX - Complete Working Solution
Run this script to fix the fire detection problem
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow/Keras imports
try:
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    print("Using TensorFlow Keras")
except ImportError:
    from keras.preprocessing import image
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.callbacks import EarlyStopping
    print("Using standalone Keras")

def load_images_simple(fire_dir, non_fire_dir, max_images=200):
    """Load images with simple preprocessing"""
    images = []
    labels = []
    
    # Load fire images
    fire_files = [f for f in os.listdir(fire_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_images//2]
    print(f"Loading {len(fire_files)} fire images...")
    
    for file in fire_files:
        try:
            img_path = os.path.join(fire_dir, file)
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(1)  # Fire = 1
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Load non-fire images
    non_fire_files = [f for f in os.listdir(non_fire_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_images//2]
    print(f"Loading {len(non_fire_files)} non-fire images...")
    
    for file in non_fire_files:
        try:
            img_path = os.path.join(non_fire_dir, file)
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(0)  # Non-fire = 0
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return np.array(images), np.array(labels)

def build_simple_model():
    """Build a simple CNN model"""
    model = Sequential([
        # Simple conv layers
        Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and dense
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def predict_fire(image_path, model, threshold=0.5):
    """Predict if image contains fire"""
    try:
        # Load image with simple preprocessing
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        
        # Make prediction
        prediction_proba = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0]
        prediction_class = 1 if prediction_proba > threshold else 0
        
        # Determine class name
        class_name = "FIRE" if prediction_class == 1 else "NON-FIRE"
        
        return {
            'image_path': image_path,
            'probability': float(prediction_proba),
            'prediction': prediction_class,
            'class_name': class_name,
            'confidence': float(abs(prediction_proba - 0.5) * 2)
        }
        
    except Exception as e:
        return {
            'image_path': image_path,
            'error': str(e),
            'probability': None,
            'prediction': None,
            'class_name': 'ERROR',
            'confidence': None
        }

def main():
    print("🔥 FIRE DETECTION FIX - COMPLETE SOLUTION")
    print("=" * 60)
    
    # Step 1: Load data
    print("Step 1: Loading data from fire_dataset folder...")
    
    fire_path = 'fire_dataset/fire_images'
    non_fire_path = 'fire_dataset/non_fire_images'
    
    # Check if paths exist
    if not os.path.exists(fire_path):
        print(f"❌ Fire path not found: {fire_path}")
        print("Available directories:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        return
    
    if not os.path.exists(non_fire_path):
        print(f"❌ Non-fire path not found: {non_fire_path}")
        return
    
    print(f"✅ Fire path found: {fire_path}")
    print(f"✅ Non-fire path found: {non_fire_path}")
    
    # Load images
    X, y = load_images_simple(fire_path, non_fire_path, max_images=200)
    
    print(f"Loaded {len(X)} images")
    print(f"Shape: {X.shape}")
    print(f"Fire images: {np.sum(y == 1)}")
    print(f"Non-fire images: {np.sum(y == 0)}")
    
    # Step 2: Split data
    print("\nStep 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Training - Fire: {np.sum(y_train == 1)}, Non-fire: {np.sum(y_train == 0)}")
    print(f"Test - Fire: {np.sum(y_test == 1)}, Non-fire: {np.sum(y_test == 0)}")
    
    # Step 3: Build model
    print("\nStep 3: Building model...")
    model = build_simple_model()
    print("Model architecture:")
    model.summary()
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"\nClass weights: {class_weight_dict}")
    print(f"  Class 0 (NON-FIRE): {class_weight_dict[0]:.3f}")
    print(f"  Class 1 (FIRE): {class_weight_dict[1]:.3f}")
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print("✅ Model training completed!")
    
    # Step 5: Test model
    print("\nStep 5: Testing model...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    print(f"\nPrediction statistics:")
    print(f"  Mean probability: {np.mean(y_pred_proba):.4f}")
    print(f"  Min probability: {np.min(y_pred_proba):.4f}")
    print(f"  Max probability: {np.max(y_pred_proba):.4f}")
    
    print(f"\nPrediction distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for pred_class, count in zip(unique, counts):
        class_name = "FIRE" if pred_class == 1 else "NON-FIRE"
        percentage = (count / len(y_pred)) * 100
        print(f"  {class_name}: {count} predictions ({percentage:.1f}%)")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.4f}")
    
    # Check if predictions are balanced
    fire_predictions = np.sum(y_pred == 1)
    fire_percentage = (fire_predictions / len(y_pred)) * 100
    
    print(f"\nPrediction balance:")
    print(f"  Fire predictions: {fire_predictions}/{len(y_pred)} ({fire_percentage:.1f}%)")
    print(f"  Non-fire predictions: {len(y_pred) - fire_predictions}/{len(y_pred)} ({100-fire_percentage:.1f}%)")
    
    # Check if model is learning correctly
    fire_prob_mean = np.mean(y_pred_proba[y_test == 1])
    non_fire_prob_mean = np.mean(y_pred_proba[y_test == 0])
    
    print(f"\nAverage prediction probabilities:")
    print(f"  Fire images: {fire_prob_mean:.4f}")
    print(f"  Non-fire images: {non_fire_prob_mean:.4f}")
    
    if fire_prob_mean > non_fire_prob_mean:
        print("✅ Model is learning correctly - fire images have higher probabilities")
    else:
        print("⚠️  Model is inverted - fire images have lower probabilities")
    
    # Step 6: Test on actual images
    print("\nStep 6: Testing on actual images from dataset...")
    
    # Get test images
    test_fire_files = [f for f in os.listdir(fire_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:2]
    test_non_fire_files = [f for f in os.listdir(non_fire_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3]
    
    print(f"Testing on {len(test_fire_files)} fire images and {len(test_non_fire_files)} non-fire images...")
    
    correct_predictions = 0
    total_predictions = 0
    
    # Test fire images
    for file in test_fire_files:
        image_path = os.path.join(fire_path, file)
        print(f"\n{'='*40}")
        print(f"Testing FIRE image: {file}")
        
        prediction_result = predict_fire(image_path, model)
        
        if 'error' not in prediction_result:
            print(f"Predicted: {prediction_result['class_name']}")
            print(f"Probability: {prediction_result['probability']:.4f}")
            print(f"Confidence: {prediction_result['confidence']:.2%}")
            
            if prediction_result['class_name'] == 'FIRE':
                print("✅ CORRECT - Fire image predicted as FIRE")
                correct_predictions += 1
            else:
                print("❌ WRONG - Fire image predicted as NON-FIRE")
            
            total_predictions += 1
        else:
            print(f"Error: {prediction_result['error']}")
    
    # Test non-fire images
    for file in test_non_fire_files:
        image_path = os.path.join(non_fire_path, file)
        print(f"\n{'='*40}")
        print(f"Testing NON-FIRE image: {file}")
        
        prediction_result = predict_fire(image_path, model)
        
        if 'error' not in prediction_result:
            print(f"Predicted: {prediction_result['class_name']}")
            print(f"Probability: {prediction_result['probability']:.4f}")
            print(f"Confidence: {prediction_result['confidence']:.2%}")
            
            if prediction_result['class_name'] == 'NON-FIRE':
                print("✅ CORRECT - Non-fire image predicted as NON-FIRE")
                correct_predictions += 1
            else:
                print("❌ WRONG - Non-fire image predicted as FIRE")
            
            total_predictions += 1
        else:
            print(f"Error: {prediction_result['error']}")
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {correct_predictions}/{total_predictions} predictions were correct")
    
    if correct_predictions == total_predictions:
        print("🎉 SUCCESS! Model is working perfectly!")
        print("   Fire images are correctly predicted as FIRE")
        print("   Non-fire images are correctly predicted as NON-FIRE")
    elif correct_predictions > total_predictions // 2:
        print("✅ GOOD! Model is working well.")
        print("   Most predictions are correct")
    else:
        print("⚠️  Model still has issues.")
        print("   This suggests a fundamental problem with the data or approach.")
    
    print("=" * 50)
    
    # Save the model
    print("\nSaving the model...")
    model.save('fire_detection_model_fixed.h5')
    print("✅ Model saved as 'fire_detection_model_fixed.h5'")
    
    print("\n🔥 FIRE DETECTION FIX COMPLETED!")
    print("You can now use the model to predict fire in new images.")

if __name__ == "__main__":
    main()
