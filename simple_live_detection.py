#!/usr/bin/env python3
"""
SIMPLE LIVE FIRE DETECTION - Easy to use live camera fire detection
"""

import cv2
import numpy as np
import os
import time

def detect_fire_color(frame):
    """Simple fire detection based on color analysis"""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for fire colors (red, orange, yellow)
    # Lower red
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    
    # Upper red
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Orange range
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    
    # Yellow range
    lower_yellow = np.array([25, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
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
    
    return fire_percentage > 5, fire_percentage, fire_mask

def detect_fire_motion(frame, background):
    """Detect fire based on motion and color"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray, bg_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for fire-colored motion
    fire_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum area
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region
            region = frame[y:y+h, x:x+w]
            
            # Check if region has fire colors
            fire_percentage, _, _ = detect_fire_color(region)
            if fire_percentage[1] > 10:  # If more than 10% fire colors
                fire_detected = True
                break
    
    return fire_detected

def main():
    """Main function for live fire detection"""
    print("🔥 SIMPLE LIVE FIRE DETECTION")
    print("=" * 40)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to reset background")
    print("=" * 40)
    
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
    print("Starting detection...")
    
    # Initialize variables
    frame_count = 0
    fire_detection_count = 0
    background = None
    background_update_interval = 30  # Update background every 30 frames
    
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
            if frame_count % background_update_interval == 0:
                background = frame.copy()
            
            # Detect fire using color analysis
            is_fire_color, fire_percentage, fire_mask = detect_fire_color(frame)
            
            # Detect fire using motion (if background is available)
            is_fire_motion = False
            if background is not None:
                is_fire_motion = detect_fire_motion(frame, background)
            
            # Combine detections
            is_fire = is_fire_color or is_fire_motion
            
            if is_fire:
                fire_detection_count += 1
            
            # Draw overlay
            overlay = frame.copy()
            
            if is_fire:
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
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Center text
            text_x = (frame.shape[1] - text_width) // 2
            text_y = (frame.shape[0] + text_height) // 2
            
            # Draw text background
            cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), 
                         (text_x + text_width + 10, text_y + 10), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
            
            # Add info text
            info_text = f"Fire %: {fire_percentage:.1f}%"
            cv2.putText(frame, info_text, (10, 30), font, 0.6, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 60), font, 0.5, (255, 255, 255), 1)
            
            # Add fire detection counter
            cv2.putText(frame, f"Fire Detections: {fire_detection_count}", (10, frame.shape[0] - 40), font, 0.5, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save, 'r' to reset", (10, frame.shape[0] - 20), font, 0.4, (255, 255, 255), 1)
            
            # Show fire mask (optional)
            if is_fire_color:
                # Resize fire mask to fit in corner
                mask_small = cv2.resize(fire_mask, (100, 100))
                mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = mask_small
            
            # Display frame
            cv2.imshow('Live Fire Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f"fire_detection_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                # Reset background
                background = frame.copy()
                print("Background reset")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")

if __name__ == "__main__":
    main()
