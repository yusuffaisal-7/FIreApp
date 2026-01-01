# 🔥 LIVE FIRE DETECTION - Real-time Camera Fire Detection

## Overview
This system provides real-time fire detection using your webcam with multiple detection methods.

## Files Available

### 1. **Python Scripts (Easiest)**
- **`simple_live_detection.py`** - Simple color-based fire detection
- **`live_fire_detection.py`** - Advanced detection with trained model

### 2. **Jupyter Notebook**
- **`live_fire_detection.ipynb`** - Interactive live detection

## 🚀 Quick Start

### Option 1: Simple Color Detection (Recommended)
```bash
python simple_live_detection.py
```

### Option 2: Advanced Detection (Requires trained model)
```bash
# First train the model
python fire_detection_fix.py

# Then run live detection
python live_fire_detection.py
```

### Option 3: Jupyter Notebook
```bash
jupyter notebook live_fire_detection.ipynb
# Run all cells in order
```

## 🎯 Features

### **Color-Based Detection**
- Detects red, orange, and yellow colors (fire colors)
- HSV color space analysis
- Configurable sensitivity

### **Motion Detection**
- Background subtraction
- Motion analysis combined with color detection
- Reduces false positives

### **Visual Alerts**
- **Red overlay** when fire is detected
- **Green overlay** when no fire
- Real-time fire percentage
- Frame counter and detection statistics

### **Controls**
- **'q'** - Quit detection
- **'s'** - Save current frame
- **'r'** - Reset background

## 🔧 How It Works

### **Color Detection Method**
1. **Convert to HSV** - Better color analysis
2. **Define fire color ranges** - Red, orange, yellow
3. **Create masks** - Isolate fire-colored pixels
4. **Calculate percentage** - Fire-colored pixels vs total
5. **Threshold detection** - Fire if > 5% fire colors

### **Motion Detection Method**
1. **Background subtraction** - Compare with previous frame
2. **Contour analysis** - Find moving objects
3. **Color verification** - Check if motion has fire colors
4. **Combine results** - Color + motion detection

## 📊 Detection Parameters

### **Color Thresholds**
- **Fire percentage**: > 5% fire-colored pixels
- **Red range**: 0-10° and 170-180°
- **Orange range**: 10-25°
- **Yellow range**: 25-35°

### **Motion Thresholds**
- **Minimum area**: 500 pixels
- **Fire color in motion**: > 10%
- **Background update**: Every 30 frames

## 🎮 Usage Instructions

1. **Start the detection**
2. **Point camera at potential fire sources**
3. **Look for red, orange, or yellow colors**
4. **System will show:**
   - Real-time fire percentage
   - Detection status (FIRE/NO FIRE)
   - Visual overlays
   - Statistics

## 🔍 Testing the System

### **Test with Fire Sources**
- Candle flames
- Lighters
- Red/orange objects
- Fire images on screen

### **Test with Non-Fire Sources**
- Regular objects
- People
- Furniture
- Non-fire colored items

## ⚠️ Important Notes

1. **Camera Access** - Make sure your camera is not being used by other applications
2. **Lighting** - Good lighting helps with detection accuracy
3. **False Positives** - Red objects may trigger false alarms
4. **Performance** - Detection runs at ~30 FPS on most systems

## 🛠️ Troubleshooting

### **Camera Not Working**
- Check if camera is connected
- Close other applications using camera
- Try different camera index (0, 1, 2)

### **Poor Detection**
- Adjust lighting conditions
- Reset background with 'r' key
- Check camera focus

### **High CPU Usage**
- Reduce frame resolution
- Increase detection interval
- Close other applications

## 📁 Output Files

- **Saved frames** - `fire_detection_frame_[timestamp].jpg`
- **Detection logs** - Console output with statistics

## 🔥 Safety Note

This is a **demonstration system** for educational purposes. For real fire safety, use professional fire detection systems and follow local safety regulations.

---

**Ready to detect fire in real-time! 🔥➡️✅**
