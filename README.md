# Fire Detection using Computer Vision and Deep Learning

A comprehensive fire detection system that uses advanced computer vision techniques and deep learning to classify images as containing fire or not. This project implements a complete pipeline from data preprocessing to model deployment.

## 🔥 Project Overview

This project implements a state-of-the-art fire detection system using:
- **Advanced Image Preprocessing** with OpenCV (masking, segmentation, sharpening)
- **Transfer Learning** with Xception pretrained model
- **Deep Learning** classifier for binary classification
- **Comprehensive Evaluation** with detailed metrics and visualizations

## 📊 Dataset

- **Total Images**: 999 images
- **Fire Images**: 755 images
- **Non-Fire Images**: 244 images
- **Image Format**: PNG files
- **Image Size**: Variable (resized to 255x255 for processing)

## 🏗️ Model Architecture

### Feature Extraction
- **Base Model**: Xception (pretrained on ImageNet)
- **Input Size**: 255x255x3
- **Feature Dimension**: 2048
- **Pooling**: Global Average Pooling

### Classifier
- **Layer 1**: Dense(256) + ReLU + Dropout(0.3)
- **Layer 2**: Dense(64) + ReLU + Dropout(0.2)
- **Output**: Dense(1) + Sigmoid
- **Total Parameters**: ~541,000

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20%

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required libraries (see requirements.txt)

### Setup
```bash
# Clone or download the project
cd fireapp

# Install required packages
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook
```

### Required Libraries
```
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
tqdm>=4.62.0
scipy>=1.7.0
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
fireapp/
├── fire_dataset/
│   ├── fire_images/          # 755 fire images
│   └── non_fire_images/      # 244 non-fire images
├── fire-detection-complete.ipynb  # Main notebook
├── fire-detection-computer-vision.ipynb  # Original notebook
├── fire_detection_model.h5   # Trained classification model
├── xception_feature_extractor.h5  # Xception feature extractor
├── fire_detection_weights.h5 # Model weights
└── README.md                # This file
```

## 🎯 Usage

### Running the Complete Pipeline

1. **Open the Notebook**:
   ```bash
   jupyter notebook fire-detection-complete.ipynb
   ```

2. **Execute All Cells**: Run all cells in sequence to:
   - Load and preprocess the dataset
   - Train the model
   - Evaluate performance
   - Test predictions

### Using the Trained Model

```python
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications import xception
import cv2

# Load the trained models
model = load_model('fire_detection_model.h5')
xception_model = load_model('xception_feature_extractor.h5')

# Function to predict fire in a new image
def predict_fire(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(255, 255))
    img_array = image.img_to_array(img)
    
    # Apply preprocessing pipeline
    image_segmented = segment_image(img_array)
    image_sharpened = sharpen_image(image_segmented)
    
    # Preprocess for Xception
    img_processed = xception.preprocess_input(np.expand_dims(image_sharpened, axis=0))
    
    # Extract features and predict
    features = xception_model.predict(img_processed)
    prediction = model.predict(features)[0][0]
    
    return "FIRE" if prediction > 0.5 else "NON-FIRE", prediction

# Example usage
result, confidence = predict_fire('path/to/your/image.jpg')
print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2%}")
```

## 🔧 Image Preprocessing Pipeline

The system applies a sophisticated preprocessing pipeline:

1. **Masking**: HSV-based color masking to isolate relevant features
2. **Segmentation**: Morphological operations for better feature extraction
3. **Sharpening**: Gaussian blur and weighted addition for detail enhancement
4. **Normalization**: Xception-specific preprocessing

## 📈 Model Performance

The model achieves high accuracy on the validation set with:
- **Accuracy**: ~89% (varies based on training)
- **Precision**: High precision for both classes
- **Recall**: Good recall for fire detection
- **F1-Score**: Balanced performance

### Training Features
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Dropout Regularization**: Prevents overfitting
- **Stratified Split**: Maintains class balance

## 🎨 Visualizations

The notebook includes comprehensive visualizations:
- **Sample Images**: Random samples from the dataset
- **Preprocessing Pipeline**: Step-by-step preprocessing visualization
- **Training History**: Loss and accuracy curves
- **Confusion Matrix**: Detailed performance metrics
- **Prediction Results**: Visual prediction examples

## 🔍 Key Features

### Advanced Preprocessing
- HSV color space masking
- Morphological operations
- Image sharpening techniques
- Robust error handling

### Transfer Learning
- Xception pretrained on ImageNet
- Feature extraction without retraining
- Efficient computation

### Model Architecture
- Custom classifier on top of Xception features
- Dropout regularization
- Binary classification with sigmoid activation

### Evaluation
- Comprehensive metrics
- Confusion matrix visualization
- Classification report
- Training history analysis

## 🚀 Deployment

### Saving Models
The notebook automatically saves:
- `fire_detection_model.h5`: Complete classification model
- `xception_feature_extractor.h5`: Xception feature extractor
- `fire_detection_weights.h5`: Model weights only

### Loading Models
```python
from keras.models import load_model

# Load complete model
model = load_model('fire_detection_model.h5')

# Load feature extractor
xception_model = load_model('xception_feature_extractor.h5')
```

## 🛠️ Customization

### Adjusting Threshold
```python
# Change classification threshold
threshold = 0.6  # Default is 0.5
prediction = "FIRE" if probability > threshold else "NON-FIRE"
```

### Modifying Architecture
```python
# Add more layers or change architecture
model = Sequential([
    Dense(512, activation='relu', input_dim=2048),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

## 📝 Notes

- **Dataset**: The model is trained on a specific dataset. Performance may vary on different datasets
- **Preprocessing**: The preprocessing pipeline is optimized for the specific image characteristics
- **Hardware**: Training requires sufficient RAM and GPU memory for optimal performance
- **Model Size**: The complete model is relatively lightweight (~2MB for classification model)

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the preprocessing pipeline
- Experimenting with different architectures
- Adding new evaluation metrics
- Optimizing performance

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Xception model architecture by François Chollet
- OpenCV for image processing
- Keras/TensorFlow for deep learning framework
- The fire detection dataset contributors

---

**Note**: This model is for educational and research purposes. For production use, ensure proper testing and validation on your specific use case.
