#!/usr/bin/env python3
"""
REAL FIRE DETECTION - Specifically designed to catch actual fire
"""

import cv2
import numpy as np
import time

def detect_fire_real(frame):
    """Detect real fire - candles, lighters, matches"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire color ranges - optimized for real fire
    # Red ranges (broader for real fire)
    lower_red1 = np.array([0, 60, 60])  # Lower saturation for real fire
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 60, 60])
    upper_red2 = np.array([180, 255, 255])
    
    # Orange range (broader)
    lower_orange = np.array([8, 60, 60])
    upper_orange = np.array([30, 255, 255])
    
    # Yellow range (broader)
    lower_yellow = np.array([20, 60, 60])
    upper_yellow = np.array([40, 255, 255])
    
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
    
    # Lower threshold to catch real fire
    is_fire = fire_percentage > 5  # Very low threshold for real fire
    
    return is_fire, fire_percentage, fire_mask

def detect_fire_brightness(frame):
    """Detect fire based on brightness and intensity"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Fire is usually bright
    if brightness > 80:  # Lower threshold for real fire
        # Check for bright spots
        _, bright_spots = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        bright_pixels = cv2.countNonZero(bright_spots)
        total_pixels = frame.shape[0] * frame.shape[1]
        bright_percentage = (bright_pixels / total_pixels) * 100
        
        return bright_percentage > 2  # Very low threshold
    
    return False

def detect_fire_flicker(frame, prev_frame):
    """Detect fire based on flickering (real fire flickers)"""
    if prev_frame is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray, prev_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    # Count flickering pixels
    flicker_pixels = cv2.countNonZero(thresh)
    total_pixels = frame.shape[0] * frame.shape[1]
    flicker_percentage = (flicker_pixels / total_pixels) * 100
    
    # Fire flickers
    return flicker_percentage > 1  # Very low threshold

def detect_fire_shape(frame):
    """Detect fire based on shape characteristics"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire color ranges
    lower_red1 = np.array([0, 60, 60])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 60, 60])
    upper_red2 = np.array([180, 255, 255])
    
    lower_orange = np.array([8, 60, 60])
    upper_orange = np.array([30, 255, 255])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine masks
    fire_mask = mask_red1 + mask_red2 + mask_orange
    
    # Find contours
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for fire-like shapes
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area for fire
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (fire is usually taller than wide)
            aspect_ratio = h / w if w > 0 else 0
            
            # Fire is usually taller than wide
            if aspect_ratio > 0.8:  # Not too strict
                # Check for upward movement (fire goes up)
                return True
    
    return False

def main():
    """Main function for real fire detection"""
    print("🔥 REAL FIRE DETECTION")
    print("=" * 50)
    print("Features:")
    print("- Detects real fire (candles, lighters, matches)")
    print("- Multiple detection methods")
    print("- Low thresholds to catch actual fire")
    print("- Flicker detection for real fire")
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
    print("Starting real fire detection...")
    print("Test with: candle, lighter, match, red objects")
    
    # Initialize variables
    frame_count = 0
    fire_detection_count = 0
    prev_frame = None
    consecutive_fire_frames = 0
    required_consecutive_frames = 2  # Only 2 consecutive frames needed
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Multiple detection methods
            is_fire_color, fire_percentage, fire_mask = detect_fire_real(frame)
            is_fire_brightness = detect_fire_brightness(frame)
            is_fire_flicker = detect_fire_flicker(frame, prev_frame)
            is_fire_shape = detect_fire_shape(frame)
            
            # Combine detections - at least 2 out of 4 methods
            methods_agree = sum([is_fire_color, is_fire_brightness, is_fire_flicker, is_fire_shape])
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
            methods_text = f"Methods: {methods_agree}/4"
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
            cv2.imshow('Real Fire Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f"real_fire_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                # Reset counters
                consecutive_fire_frames = 0
                fire_detection_count = 0
                frame_count = 0
                prev_frame = None
                print("Reset complete")
            
            # Update previous frame
            prev_frame = frame.copy()
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
