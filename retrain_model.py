#!/usr/bin/env python3
"""
Quick script to retrain the fire detection model with correct labels.
Run this if you're getting incorrect predictions.
"""

import os
import sys

def main():
    print("🔥 FIRE DETECTION MODEL RETRAINING SCRIPT")
    print("=" * 50)
    print()
    print("If you're getting incorrect predictions (forest showing as fire),")
    print("follow these steps:")
    print()
    print("1. Open the Jupyter notebook: fire-detection-complete.ipynb")
    print("2. Go to Kernel → Restart & Clear Output")
    print("3. Run all cells from the beginning")
    print("4. The model will be retrained with correct labels")
    print()
    print("IMPORTANT LABEL MAPPING:")
    print("  - fire_images → ID = 1 (FIRE)")
    print("  - non_fire_images → ID = 0 (NON-FIRE)")
    print()
    print("The notebook now includes verification steps to ensure")
    print("labels are correctly assigned before training.")
    print()
    print("After retraining, forest images should correctly predict as NON-FIRE")
    print("and fire images should correctly predict as FIRE.")
    print()
    print("=" * 50)

if __name__ == "__main__":
    main()
