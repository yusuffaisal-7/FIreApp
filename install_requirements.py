#!/usr/bin/env python3
"""
Install required packages for fire detection
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

def main():
    print("📦 INSTALLING REQUIRED PACKAGES")
    print("=" * 40)
    
    packages = [
        "tensorflow",
        "numpy",
        "pandas", 
        "matplotlib",
        "scikit-learn",
        "opencv-python",
        "pillow"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        install_package(package)
    
    print("\n✅ All packages installed!")
    print("You can now run the fire detection script.")

if __name__ == "__main__":
    main()
