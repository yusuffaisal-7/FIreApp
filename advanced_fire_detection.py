#!/usr/bin/env python3
"""
ADVANCED FIRE DETECTION - Advanced filtering to exclude screens and monitors
"""

import cv2
import numpy as np
import time

def detect_screen_monitor(frame):
    """Advanced screen/monitor detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find bright areas
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = np.ones((5,5), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Large bright area
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (screens are rectangular)
            aspect_ratio = w / h if h > 0 else 0
            
            # Screens typically have aspect ratios between 1.2 and 3.0
            if 1.2 <= aspect_ratio <= 3.0:
                # Check uniformity (screens are very uniform)
                region = gray[y:y+h, x:x+w]
                if region.size > 0:
                    # Calculate standard deviation
                    std_dev = np.std(region)
                    
                    # Screens are very uniform (low standard deviation)
                    if std_dev < 25:
                        # Check for rectangular shape
                        contour_area = cv2.contourArea(contour)
                        bbox_area = w * h
                        if bbox_area > 0:
                            fill_ratio = contour_area / bbox_area
                            
                            # Screens have high fill ratio (rectangular)
                            if fill_ratio > 0.8:
                                return True, (x, y, w, h)
    
    return False, None

def detect_fire_color_advanced(frame):
    """Advanced fire color detection with screen filtering"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire color ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    lower_orange = np.array([8, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    fire_mask = mask_red1 + mask_red2 + mask_orange + mask_yellow
    
    # Check for screen
    is_screen, screen_bbox = detect_screen_monitor(frame)
    
    if is_screen:
        # If screen detected, exclude that area from fire detection
        x, y, w, h = screen_bbox
        fire_mask[y:y+h, x:x+w] = 0  # Remove screen area from fire mask
    
    # Count fire pixels
    fire_pixels = cv2.countNonZero(fire_mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    fire_percentage = (fire_pixels / total_pixels) * 100
    
    # Fire detection threshold
    is_fire = fire_percentage > 5
    
    return is_fire, fire_percentage, fire_mask, is_screen

def detect_fire_motion_advanced(frame, prev_frame):
    """Advanced motion detection with screen filtering"""
    if prev_frame is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray, prev_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Check for screen
    is_screen, screen_bbox = detect_screen_monitor(frame)
    
    if is_screen:
        # If screen detected, exclude that area from motion detection
        x, y, w, h = screen_bbox
        thresh[y:y+h, x:x+w] = 0  # Remove screen area from motion mask
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for motion with fire colors
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region
            region = frame[y:y+h, x:x+w]
            
            # Check if region has fire colors
            _, fire_percentage, _, _ = detect_fire_color_advanced(region)
            if fire_percentage > 10:
                return True
    
    return False

def detect_fire_flicker_advanced(frame, prev_frame):
    """Advanced flicker detection - real fire flickers, screens don't"""
    if prev_frame is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray, prev_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Check for screen
    is_screen, screen_bbox = detect_screen_monitor(frame)
    
    if is_screen:
        # If screen detected, exclude that area from flicker detection
        x, y, w, h = screen_bbox
        thresh[y:y+h, x:x+w] = 0  # Remove screen area from flicker mask
    
    # Count flickering pixels
    flicker_pixels = cv2.countNonZero(thresh)
    total_pixels = frame.shape[0] * frame.shape[1]
    flicker_percentage = (flicker_pixels / total_pixels) * 100
    
    # Real fire flickers more than screens
    return flicker_percentage > 3

def detect_fire_shape_advanced(frame):
    """Advanced shape detection - fire has irregular shapes"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire color ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    lower_orange = np.array([8, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine masks
    fire_mask = mask_red1 + mask_red2 + mask_orange
    
    # Check for screen
    is_screen, screen_bbox = detect_screen_monitor(frame)
    
    if is_screen:
        # If screen detected, exclude that area from shape detection
        x, y, w, h = screen_bbox
        fire_mask[y:y+h, x:x+w] = 0  # Remove screen area from fire mask
    
    # Find contours
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate contour area vs bounding box area
            bbox_area = w * h
            if bbox_area > 0:
                fill_ratio = area / bbox_area
                
                # Fire has irregular shapes (low fill ratio)
                if fill_ratio < 0.6:
                    return True
    
    return False

def main():
    """Main function for advanced fire detection"""
    print("🔥 ADVANCED FIRE DETECTION")
    print("=" * 50)
    print("Features:")
    print("- Advanced screen/monitor filtering")
    print("- Multiple detection methods")
    print("- Screen area exclusion")
    print("- Irregular shape detection")
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
    print("Starting advanced detection...")
    print("Test with: candle, lighter, laptop screen, monitor")
    
    # Initialize variables
    frame_count = 0
    fire_detection_count = 0
    prev_frame = None
    consecutive_fire_frames = 0
    required_consecutive_frames = 3
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Advanced detection methods
            is_fire_color, fire_percentage, fire_mask, is_screen = detect_fire_color_advanced(frame)
            is_fire_motion = detect_fire_motion_advanced(frame, prev_frame)
            is_fire_flicker = detect_fire_flicker_advanced(frame, prev_frame)
            is_fire_shape = detect_fire_shape_advanced(frame)
            
            # Combine detections
            methods_agree = sum([is_fire_color, is_fire_motion, is_fire_flicker, is_fire_shape])
            
            # If screen detected, require more methods to agree
            if is_screen:
                is_fire = methods_agree >= 3  # Need 3 out of 4 methods
            else:
                is_fire = methods_agree >= 2  # Need 2 out of 4 methods
            
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
            
            # Add screen detection info
            screen_text = f"Screen: {'Yes' if is_screen else 'No'}"
            cv2.putText(frame, screen_text, (10, 90), font, 0.6, (255, 255, 255), 2)
            
            # Add consecutive frames
            consecutive_text = f"Consecutive: {consecutive_fire_frames}/{required_consecutive_frames}"
            cv2.putText(frame, consecutive_text, (10, 120), font, 0.6, (255, 255, 255), 2)
            
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
            cv2.imshow('Advanced Fire Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f"advanced_fire_detection_{timestamp}.jpg"
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
