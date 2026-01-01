#!/usr/bin/env python3
"""
ACCURATE LIVE FIRE DETECTION - Reduced False Positives
"""

import cv2
import numpy as np
import time
import os

class AccurateFireDetector:
    def __init__(self):
        """Initialize accurate fire detector with strict parameters"""
        self.fire_detection_count = 0
        self.total_frames = 0
        self.consecutive_fire_frames = 0
        self.required_consecutive_frames = 5  # Require 5 consecutive frames for fire
        self.background = None
        self.background_update_interval = 60  # Update background less frequently
        
    def detect_fire_color_strict(self, frame):
        """Strict fire detection with higher thresholds"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # More restrictive fire color ranges
        # Red range (more restrictive)
        lower_red1 = np.array([0, 100, 100])  # Higher saturation and value
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Orange range (more restrictive)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([20, 255, 255])
        
        # Yellow range (more restrictive)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        fire_mask = mask_red1 + mask_red2 + mask_orange + mask_yellow
        
        # Count fire-colored pixels
        fire_pixels = cv2.countNonZero(fire_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_percentage = (fire_pixels / total_pixels) * 100
        
        # Higher threshold for fire detection
        is_fire = fire_percentage > 15  # Increased from 5% to 15%
        
        return is_fire, fire_percentage, fire_mask
    
    def detect_fire_motion_strict(self, frame):
        """Strict motion-based fire detection"""
        if self.background is None:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray, bg_gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)  # Higher threshold
        
        # Morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for fire-colored motion with stricter criteria
        fire_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Higher minimum area
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract region
                region = frame[y:y+h, x:x+w]
                
                # Check if region has fire colors with higher threshold
                _, fire_percentage, _ = self.detect_fire_color_strict(region)
                if fire_percentage > 20:  # Higher threshold for motion detection
                    fire_detected = True
                    break
        
        return fire_detected
    
    def detect_fire_shape(self, frame):
        """Detect fire-like shapes and patterns"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for fire-like shapes (tall, irregular)
        fire_shapes = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (fire is usually taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                
                # Check for irregular shape (fire has irregular edges)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
                    
                    # Fire-like characteristics
                    if aspect_ratio > 1.2 and circularity < 0.3:  # Tall and irregular
                        fire_shapes += 1
        
        return fire_shapes > 2  # Require multiple fire-like shapes
    
    def detect_fire_combined(self, frame):
        """Combined fire detection with multiple methods"""
        # Color detection
        is_fire_color, fire_percentage, fire_mask = self.detect_fire_color_strict(frame)
        
        # Motion detection
        is_fire_motion = self.detect_fire_motion_strict(frame)
        
        # Shape detection
        is_fire_shape = self.detect_fire_shape(frame)
        
        # Combine results with strict requirements
        # Require at least 2 out of 3 methods to agree
        methods_agree = sum([is_fire_color, is_fire_motion, is_fire_shape])
        
        # Also check for consecutive frames
        if methods_agree >= 2:
            self.consecutive_fire_frames += 1
        else:
            self.consecutive_fire_frames = 0
        
        # Only confirm fire if multiple methods agree AND consecutive frames
        is_fire = (methods_agree >= 2) and (self.consecutive_fire_frames >= self.required_consecutive_frames)
        
        return is_fire, fire_percentage, fire_mask, methods_agree
    
    def draw_overlay_accurate(self, frame, is_fire, fire_percentage, methods_agree):
        """Draw accurate overlay with more information"""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        if is_fire:
            # Fire detected - Red overlay
            color = (0, 0, 255)  # Red
            text = "🔥 FIRE DETECTED!"
            text_color = (255, 255, 255)  # White
            
            # Draw red rectangle
            cv2.rectangle(overlay, (0, 0), (width, height), color, -1)
            
            # Add transparency
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
        else:
            # No fire - Green overlay
            color = (0, 255, 0)  # Green
            text = "✅ NO FIRE"
            text_color = (0, 0, 0)  # Black
        
        # Draw border
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), color, 3)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
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
        
        # Add detailed info
        info_text = f"Fire %: {fire_percentage:.1f}%"
        cv2.putText(frame, info_text, (10, 30), font, 0.6, (255, 255, 255), 2)
        
        # Add methods agreement
        methods_text = f"Methods: {methods_agree}/3"
        cv2.putText(frame, methods_text, (10, 60), font, 0.6, (255, 255, 255), 2)
        
        # Add consecutive frames
        consecutive_text = f"Consecutive: {self.consecutive_fire_frames}/{self.required_consecutive_frames}"
        cv2.putText(frame, consecutive_text, (10, 90), font, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.total_frames}", (10, height - 80), font, 0.5, (255, 255, 255), 1)
        
        # Add fire detection counter
        cv2.putText(frame, f"Fire Detections: {self.fire_detection_count}", (10, height - 60), font, 0.5, (255, 255, 255), 1)
        
        # Add accuracy info
        accuracy_text = f"Accuracy: {(self.total_frames - self.fire_detection_count) / max(self.total_frames, 1) * 100:.1f}%"
        cv2.putText(frame, accuracy_text, (10, height - 40), font, 0.5, (255, 255, 255), 1)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save, 'r' to reset", (10, height - 20), font, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def start_detection(self):
        """Start accurate fire detection"""
        print("🔥 ACCURATE LIVE FIRE DETECTION")
        print("=" * 50)
        print("Features:")
        print("- Higher thresholds to reduce false positives")
        print("- Multiple detection methods")
        print("- Consecutive frame requirement")
        print("- Shape analysis")
        print("=" * 50)
        print("Press 'q' to quit, 's' to save, 'r' to reset")
        print("=" * 50)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully!")
        print("Starting accurate detection...")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Update background periodically
                if self.total_frames % self.background_update_interval == 0:
                    self.background = frame.copy()
                
                # Detect fire with combined methods
                is_fire, fire_percentage, fire_mask, methods_agree = self.detect_fire_combined(frame)
                
                if is_fire:
                    self.fire_detection_count += 1
                
                # Draw overlay
                frame_with_overlay = self.draw_overlay_accurate(frame, is_fire, fire_percentage, methods_agree)
                
                # Display frame
                cv2.imshow('Accurate Live Fire Detection', frame_with_overlay)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"accurate_fire_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_with_overlay)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset background and counters
                    self.background = frame.copy()
                    self.consecutive_fire_frames = 0
                    print("Background and counters reset")
                
                self.total_frames += 1
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Detection stopped")
            print(f"Total frames: {self.total_frames}")
            print(f"Fire detections: {self.fire_detection_count}")
            print(f"Accuracy: {(self.total_frames - self.fire_detection_count) / max(self.total_frames, 1) * 100:.1f}%")

def main():
    """Main function"""
    detector = AccurateFireDetector()
    detector.start_detection()

if __name__ == "__main__":
    main()
