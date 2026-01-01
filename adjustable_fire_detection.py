#!/usr/bin/env python3
"""
ADJUSTABLE FIRE DETECTION - Control sensitivity to reduce false positives
"""

import cv2
import numpy as np
import time

class AdjustableFireDetector:
    def __init__(self):
        """Initialize with adjustable parameters"""
        # Adjustable parameters
        self.fire_threshold = 20  # Higher = less sensitive
        self.motion_threshold = 1000  # Higher = less sensitive
        self.consecutive_frames = 3  # Require consecutive frames
        self.color_sensitivity = 100  # Higher = more restrictive
        
        # Counters
        self.fire_detections = 0
        self.total_frames = 0
        self.consecutive_fire = 0
        self.background = None
        
    def detect_fire_adjustable(self, frame):
        """Fire detection with adjustable sensitivity"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjustable color ranges based on sensitivity
        lower_red1 = np.array([0, self.color_sensitivity, self.color_sensitivity])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, self.color_sensitivity, self.color_sensitivity])
        upper_red2 = np.array([180, 255, 255])
        
        lower_orange = np.array([10, self.color_sensitivity, self.color_sensitivity])
        upper_orange = np.array([20, 255, 255])
        
        lower_yellow = np.array([20, self.color_sensitivity, self.color_sensitivity])
        upper_yellow = np.array([30, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        fire_mask = mask_red1 + mask_red2 + mask_orange + mask_yellow
        
        # Count fire pixels
        fire_pixels = cv2.countNonZero(fire_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_percentage = (fire_pixels / total_pixels) * 100
        
        # Check if fire detected
        is_fire = fire_percentage > self.fire_threshold
        
        return is_fire, fire_percentage, fire_mask
    
    def detect_motion_adjustable(self, frame):
        """Motion detection with adjustable sensitivity"""
        if self.background is None:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray, bg_gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                return True
        
        return False
    
    def draw_interface(self, frame, is_fire, fire_percentage):
        """Draw interface with controls"""
        height, width = frame.shape[:2]
        
        # Draw border
        color = (0, 0, 255) if is_fire else (0, 255, 0)
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), color, 3)
        
        # Add status text
        status = "🔥 FIRE DETECTED!" if is_fire else "✅ NO FIRE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, status, (20, 40), font, 1, color, 2)
        
        # Add fire percentage
        cv2.putText(frame, f"Fire %: {fire_percentage:.1f}%", (20, 80), font, 0.7, (255, 255, 255), 2)
        
        # Add threshold info
        cv2.putText(frame, f"Threshold: {self.fire_threshold}%", (20, 110), font, 0.6, (255, 255, 255), 2)
        
        # Add consecutive frames
        cv2.putText(frame, f"Consecutive: {self.consecutive_fire}/{self.consecutive_frames}", (20, 140), font, 0.6, (255, 255, 255), 2)
        
        # Add controls
        cv2.putText(frame, "Controls:", (20, height - 120), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Q=Quit, S=Save, R=Reset", (20, height - 100), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "1/2=Threshold, 3/4=Motion, 5/6=Color", (20, height - 80), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "7/8=Consecutive frames", (20, height - 60), font, 0.5, (255, 255, 255), 1)
        
        # Add statistics
        cv2.putText(frame, f"Detections: {self.fire_detections}", (20, height - 40), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames: {self.total_frames}", (20, height - 20), font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_detection(self):
        """Start adjustable fire detection"""
        print("🔥 ADJUSTABLE FIRE DETECTION")
        print("=" * 50)
        print("Controls:")
        print("1/2 - Adjust fire threshold (higher = less sensitive)")
        print("3/4 - Adjust motion threshold")
        print("5/6 - Adjust color sensitivity")
        print("7/8 - Adjust consecutive frames required")
        print("Q - Quit, S - Save, R - Reset")
        print("=" * 50)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Camera initialized!")
        print("Starting detection...")
        print("Use number keys to adjust sensitivity!")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Update background every 30 frames
                if self.total_frames % 30 == 0:
                    self.background = frame.copy()
                
                # Detect fire
                is_fire_color, fire_percentage, fire_mask = self.detect_fire_adjustable(frame)
                is_fire_motion = self.detect_motion_adjustable(frame)
                
                # Combine detections
                is_fire = is_fire_color and is_fire_motion
                
                # Check consecutive frames
                if is_fire:
                    self.consecutive_fire += 1
                else:
                    self.consecutive_fire = 0
                
                # Only confirm fire after consecutive frames
                confirmed_fire = self.consecutive_fire >= self.consecutive_frames
                
                if confirmed_fire:
                    self.fire_detections += 1
                
                # Draw interface
                frame_with_interface = self.draw_interface(frame, confirmed_fire, fire_percentage)
                
                # Display frame
                cv2.imshow('Adjustable Fire Detection', frame_with_interface)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save frame
                    timestamp = int(time.time())
                    filename = f"fire_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_with_interface)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset
                    self.background = frame.copy()
                    self.consecutive_fire = 0
                    self.fire_detections = 0
                    self.total_frames = 0
                    print("Reset complete")
                elif key == ord('1'):
                    # Decrease fire threshold (more sensitive)
                    self.fire_threshold = max(5, self.fire_threshold - 5)
                    print(f"Fire threshold: {self.fire_threshold}%")
                elif key == ord('2'):
                    # Increase fire threshold (less sensitive)
                    self.fire_threshold = min(50, self.fire_threshold + 5)
                    print(f"Fire threshold: {self.fire_threshold}%")
                elif key == ord('3'):
                    # Decrease motion threshold (more sensitive)
                    self.motion_threshold = max(100, self.motion_threshold - 100)
                    print(f"Motion threshold: {self.motion_threshold}")
                elif key == ord('4'):
                    # Increase motion threshold (less sensitive)
                    self.motion_threshold = min(5000, self.motion_threshold + 100)
                    print(f"Motion threshold: {self.motion_threshold}")
                elif key == ord('5'):
                    # Decrease color sensitivity (more sensitive)
                    self.color_sensitivity = max(50, self.color_sensitivity - 10)
                    print(f"Color sensitivity: {self.color_sensitivity}")
                elif key == ord('6'):
                    # Increase color sensitivity (less sensitive)
                    self.color_sensitivity = min(200, self.color_sensitivity + 10)
                    print(f"Color sensitivity: {self.color_sensitivity}")
                elif key == ord('7'):
                    # Decrease consecutive frames
                    self.consecutive_frames = max(1, self.consecutive_frames - 1)
                    print(f"Consecutive frames: {self.consecutive_frames}")
                elif key == ord('8'):
                    # Increase consecutive frames
                    self.consecutive_frames = min(10, self.consecutive_frames + 1)
                    print(f"Consecutive frames: {self.consecutive_frames}")
                
                self.total_frames += 1
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Detection stopped")
            print(f"Total frames: {self.total_frames}")
            print(f"Fire detections: {self.fire_detections}")

def main():
    """Main function"""
    detector = AdjustableFireDetector()
    detector.start_detection()

if __name__ == "__main__":
    main()
