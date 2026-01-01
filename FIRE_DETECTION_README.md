# 🔥 FIRE DETECTION SYSTEMS

## Overview
This project contains multiple fire detection systems designed for different use cases. Each system has different sensitivity levels and detection methods.

## 🚀 Quick Start

### Test All Systems
```bash
python test_fire_detection.py
```

### Individual Systems
```bash
# Balanced detection (recommended)
python balanced_fire_detection.py

# Real fire detection (catches actual fire)
python real_fire_detection.py

# Less sensitive (fewer false positives)
python less_sensitive_detection.py

# Adjustable sensitivity
python adjustable_fire_detection.py
```

## 📋 Available Systems

### 1. **Balanced Fire Detection** (`balanced_fire_detection.py`)
- **Best for**: General use
- **Features**: 
  - Can detect real fire (candles, lighters)
  - Reduces false positives from red objects
  - Multiple detection methods
  - Balanced thresholds
- **Sensitivity**: Medium
- **False Positives**: Low
- **Real Fire Detection**: Good

### 2. **Real Fire Detection** (`real_fire_detection.py`)
- **Best for**: Catching actual fire
- **Features**:
  - Specifically designed for real fire
  - Flicker detection (real fire flickers)
  - Shape analysis
  - Very low thresholds
- **Sensitivity**: High
- **False Positives**: Medium
- **Real Fire Detection**: Excellent

### 3. **Less Sensitive Detection** (`less_sensitive_detection.py`)
- **Best for**: Reducing false positives
- **Features**:
  - High thresholds
  - Multiple confirmation methods
  - Strict requirements
- **Sensitivity**: Low
- **False Positives**: Very Low
- **Real Fire Detection**: Poor

### 4. **Adjustable Detection** (`adjustable_fire_detection.py`)
- **Best for**: Customizable sensitivity
- **Features**:
  - User-adjustable sensitivity
  - Real-time threshold adjustment
  - Multiple detection methods
- **Sensitivity**: Adjustable
- **False Positives**: Adjustable
- **Real Fire Detection**: Adjustable

## 🎯 Which System to Use?

### For Real Fire Detection (Candles, Lighters, Matches)
```bash
python real_fire_detection.py
```

### For Balanced Performance
```bash
python balanced_fire_detection.py
```

### For Fewer False Positives
```bash
python less_sensitive_detection.py
```

### For Customizable Detection
```bash
python adjustable_fire_detection.py
```

## 🔧 Controls

All systems support the same controls:
- **'q'**: Quit
- **'s'**: Save current frame
- **'r'**: Reset detection

## 📊 Performance Comparison

| System | Real Fire Detection | False Positives | Speed | Accuracy |
|--------|-------------------|-----------------|-------|----------|
| Balanced | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Real Fire | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Less Sensitive | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Adjustable | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🧪 Testing

### Test with Real Fire
1. Light a candle
2. Use a lighter
3. Light a match
4. Test with different lighting conditions

### Test with Red Objects
1. Red clothing
2. Red objects
3. Red lights
4. Red screens

### Test with Different Lighting
1. Bright sunlight
2. Dim lighting
3. Artificial lighting
4. Mixed lighting

## 🔍 Troubleshooting

### "Not detecting real fire"
- Use `real_fire_detection.py`
- Check lighting conditions
- Ensure fire is visible in camera
- Try different angles

### "Too many false positives"
- Use `less_sensitive_detection.py`
- Adjust sensitivity in `adjustable_fire_detection.py`
- Check for red objects in background

### "System not working"
- Check camera permissions
- Ensure OpenCV is installed
- Try different camera (if available)

## 📈 Optimization Tips

1. **Lighting**: Ensure good lighting for better detection
2. **Distance**: Keep fire at appropriate distance from camera
3. **Background**: Avoid red backgrounds
4. **Movement**: Keep camera steady for better detection
5. **Testing**: Test with different fire sources

## 🛠️ Technical Details

### Detection Methods
- **Color Detection**: HSV color space analysis
- **Motion Detection**: Frame difference analysis
- **Brightness Analysis**: Intensity-based detection
- **Shape Analysis**: Contour and aspect ratio analysis
- **Flicker Detection**: Temporal analysis for real fire

### Thresholds
- **Fire Percentage**: 5-15% of frame
- **Brightness**: 80-200 intensity
- **Motion**: 800+ pixel area
- **Consecutive Frames**: 2-5 frames required

## 📝 Notes

- All systems work in real-time
- Detection is based on computer vision
- No machine learning models required
- Works with any camera
- Cross-platform compatible

## 🚨 Safety Note

These systems are for demonstration and testing purposes only. They should not be used as the sole fire detection system in critical applications. Always use proper fire safety equipment and systems.
