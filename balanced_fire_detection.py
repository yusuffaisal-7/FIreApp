#!/usr/bin/env python3
"""
BALANCED FIRE DETECTION - Can detect real fire while reducing false positives
"""

import cv2
import numpy as np
import time

def detect_fire_balanced(frame):
    """Balanced fire detection - can detect real fire"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Balanced fire color ranges - not too strict, not too loose
    # Red ranges
    lower_red1 = np.array([0, 80, 80])  # Moderate saturation and value
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])
    
    # Orange range
    lower_orange = np.array([10, 80, 80])
    upper_orange = np.array([25, 255, 255])
    
    # Yellow range
    lower_yellow = np.array([25, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    
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
    
    # Balanced threshold - can detect real fire
    is_fire = fire_percentage > 8  # Lower threshold to catch real fire
    
    return is_fire, fire_percentage, fire_mask

def detect_fire_motion_balanced(frame, background):
    """Balanced motion detection"""
    if background is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray, bg_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to reduce noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for motion with fire colors
    for contour in contours:
        if cv2.contourArea(contour) > 800:  # Moderate area requirement
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region
            region = frame[y:y+h, x:x+w]
            
            # Check if region has fire colors
            _, fire_percentage, _ = detect_fire_balanced(region)
            if fire_percentage > 12:  # Moderate threshold
                return True
    
    return False

def detect_fire_intensity(frame):
    """Detect fire based on intensity and brightness"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Fire is usually bright
    if brightness > 100:  # Bright enough for fire
        # Check for bright spots
        _, bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_pixels = cv2.countNonZero(bright_spots)
        total_pixels = frame.shape[0] * frame.shape[1]
        bright_percentage = (bright_pixels / total_pixels) * 100
        
        return bright_percentage > 3  # Some bright spots
    
    return False

def main():
    """Main function for balanced fire detection"""
    print("🔥 BALANCED FIRE DETECTION")
    print("=" * 50)
    print("Features:")
    print("- Can detect real fire (candles, lighters)")
    print("- Reduces false positives from red objects")
    print("- Multiple detection methods")
    print("- Balanced thresholds")
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
    
    print("✅ Camera initialized!")
    print("Starting balanced detection...")
    print("Test with: candle, lighter, red objects")
    
    # Initialize variables
    frame_count = 0
    fire_detection_count = 0
    background = None
    consecutive_fire_frames = 0
    required_consecutive_frames = 3  # Require 3 consecutive frames
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Update background every 30 frames
            if frame_count % 30 == 0:
                background = frame.copy()
            
            # Multiple detection methods
            is_fire_color, fire_percentage, fire_mask = detect_fire_balanced(frame)
            is_fire_motion = detect_fire_motion_balanced(frame, background)
            is_fire_intensity = detect_fire_intensity(frame)
            
            # Combine detections - at least 2 out of 3 methods
            methods_agree = sum([is_fire_color, is_fire_motion, is_fire_intensity])
            is_fire = methods_agree >= 2
            
            # Check consecutive frames
            if is_fire:
                consecutive_fire_frames += 1
            else:
                consecutive_fire_frames = 0
            
            # Only confirm fire after consecutive frames
            confirmed_fire = consecutive_fire_frames >= required_consecutive_frames
            
            if confirmed_fire:
                fire_detection_count += 1
            
            # Draw overlay
            overlay = frame.copy()
            
            if confirmed_fire:
                # Fire detected - Red overlay
                color = (0, 0, 255)  # Red
                text = "🔥 FIRE DETECTED!"
                text_color = (255, 255, 255)  # White
                
                # Draw red rectangle
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
                
                # Add transparency
                alpha = 0.3
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
            else:
                # No fire - Green overlay
                color = (0, 255, 0)  # Green
                text = "✅ NO FIRE"
                text_color = (0, 0, 0)  # Black
            
            # Draw border
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), color, 3)
            
            # Add text to right side top
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position text at right side top
            text_x = frame.shape[1] - text_width - 20  # 20 pixels from right edge
            text_y = 40  # 40 pixels from top
            
            # Draw text background
            cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), 
                         (text_x + text_width + 10, text_y + 10), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
            
            # Add detailed info
            info_text = f"Fire %: {fire_percentage:.1f}%"
            cv2.putText(frame, info_text, (10, 30), font, 0.6, (255, 255, 255), 2)
            
            # Add methods info
            methods_text = f"Methods: {methods_agree}/3"
            cv2.putText(frame, methods_text, (10, 60), font, 0.6, (255, 255, 255), 2)
            
            # Add consecutive frames
            consecutive_text = f"Consecutive: {consecutive_fire_frames}/{required_consecutive_frames}"
            cv2.putText(frame, consecutive_text, (10, 90), font, 0.6, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 80), font, 0.5, (255, 255, 255), 1)
            
            # Add fire detection counter
            cv2.putText(frame, f"Fire Detections: {fire_detection_count}", (10, frame.shape[0] - 60), font, 0.5, (255, 255, 255), 1)
            
            # Add accuracy info
            accuracy = (frame_count - fire_detection_count) / max(frame_count, 1) * 100
            cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (10, frame.shape[0] - 40), font, 0.5, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save, 'r' to reset", (10, frame.shape[0] - 20), font, 0.4, (255, 255, 255), 1)
            
            # Show fire mask in corner
            if is_fire_color:
                mask_small = cv2.resize(fire_mask, (100, 100))
                mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = mask_small
            
            # Display frame
            cv2.imshow('Balanced Fire Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f"balanced_fire_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                # Reset background and counters
                background = frame.copy()
                consecutive_fire_frames = 0
                fire_detection_count = 0
                frame_count = 0
                print("Reset complete")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")
        print(f"Total frames: {frame_count}")
        print(f"Fire detections: {fire_detection_count}")
        print(f"Accuracy: {(frame_count - fire_detection_count) / max(frame_count, 1) * 100:.1f}%")

if __name__ == "__main__":
    main()
