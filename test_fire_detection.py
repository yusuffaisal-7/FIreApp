#!/usr/bin/env python3
"""
TEST FIRE DETECTION - Simple test script
"""

import cv2
import numpy as np
import time

def test_fire_detection():
    """Test fire detection with different methods"""
    print("🔥 FIRE DETECTION TEST")
    print("=" * 50)
    print("Available tests:")
    print("1. Balanced Detection (recommended)")
    print("2. Real Fire Detection (catches actual fire)")
    print("3. Less Sensitive Detection (fewer false positives)")
    print("4. Adjustable Detection (customizable)")
    print("5. Smart Detection (filters screens)")
    print("6. Advanced Detection (best for screens)")
    print("=" * 50)
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == "1":
        print("Starting Balanced Detection...")
        import subprocess
        subprocess.run(["python", "balanced_fire_detection.py"])
    elif choice == "2":
        print("Starting Real Fire Detection...")
        import subprocess
        subprocess.run(["python", "real_fire_detection.py"])
    elif choice == "3":
        print("Starting Less Sensitive Detection...")
        import subprocess
        subprocess.run(["python", "less_sensitive_detection.py"])
    elif choice == "4":
        print("Starting Adjustable Detection...")
        import subprocess
        subprocess.run(["python", "adjustable_fire_detection.py"])
    elif choice == "5":
        print("Starting Smart Detection...")
        import subprocess
        subprocess.run(["python", "smart_fire_detection.py"])
    elif choice == "6":
        print("Starting Advanced Detection...")
        import subprocess
        subprocess.run(["python", "advanced_fire_detection.py"])
    else:
        print("Invalid choice. Please run again and select 1-6.")

if __name__ == "__main__":
    test_fire_detection()
