#!/usr/bin/env python3
"""
LIVE FIRE DETECTION - Real-time camera fire detection
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import os

class LiveFireDetector:
    def __init__(self, model_path='fire_detection_model_fixed.h5'):
        """Initialize the live fire detector"""
        self.model_path = model_path
        self.model = None
        self.cap = None
        self.load_model()
        
    def load_model(self):
        """Load the trained fire detection model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.model = load_model(self.model_path)
                print("✅ Model loaded successfully!")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("Please train the model first using fire_detection_fix.py")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        # Resize frame to match model input size
        frame_resized = cv2.resize(frame, (150, 150))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict_fire(self, frame):
        """Predict if frame contains fire"""
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Make prediction
            prediction_proba = self.model.predict(processed_frame, verbose=0)[0][0]
            
            # Determine if fire is detected
            is_fire = prediction_proba > 0.5
            
            return is_fire, prediction_proba
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return False, 0.0
    
    def draw_overlay(self, frame, is_fire, confidence):
        """Draw overlay on frame"""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        if is_fire:
            # Fire detected - Red overlay
            color = (0, 0, 255)  # Red in BGR
            text = "🔥 FIRE DETECTED!"
            text_color = (255, 255, 255)  # White text
            
            # Draw red rectangle
            cv2.rectangle(overlay, (0, 0), (width, height), color, -1)
            
            # Add transparency
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
        else:
            # No fire - Green overlay
            color = (0, 255, 0)  # Green in BGR
            text = "✅ NO FIRE"
            text_color = (0, 0, 0)  # Black text
        
        # Draw border
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), color, 3)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Center text
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2
        
        # Draw text background
        cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), 
                     (text_x + text_width + 10, text_y + 10), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Add confidence score
        confidence_text = f"Confidence: {confidence:.2%}"
        cv2.putText(frame, confidence_text, (10, 30), font, 0.7, text_color, 2)
        
        return frame
    
    def start_detection(self):
        """Start live fire detection"""
        print("🔥 STARTING LIVE FIRE DETECTION")
        print("=" * 40)
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("=" * 40)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully!")
        print("Starting detection...")
        
        frame_count = 0
        fire_detection_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Predict every 5th frame for better performance
                if frame_count % 5 == 0:
                    is_fire, confidence = self.predict_fire(frame)
                    if is_fire:
                        fire_detection_count += 1
                else:
                    # Use previous prediction
                    pass
                
                # Draw overlay
                frame_with_overlay = self.draw_overlay(frame, is_fire, confidence)
                
                # Add frame counter
                cv2.putText(frame_with_overlay, f"Frame: {frame_count}", 
                           (10, frame_with_overlay.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add fire detection counter
                cv2.putText(frame_with_overlay, f"Fire Detections: {fire_detection_count}", 
                           (10, frame_with_overlay.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Live Fire Detection', frame_with_overlay)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"fire_detection_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_with_overlay)
                    print(f"Frame saved as {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Detection stopped")

def main():
    """Main function"""
    print("🔥 LIVE FIRE DETECTION SYSTEM")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'fire_detection_model_fixed.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first:")
        print("1. Run: python fire_detection_fix.py")
        print("2. Or use the Jupyter notebook: fire_detection_simple.ipynb")
        return
    
    # Create detector
    detector = LiveFireDetector(model_path)
    
    if detector.model is None:
        print("❌ Failed to load model")
        return
    
    # Start detection
    detector.start_detection()

if __name__ == "__main__":
    main()
