# Sign Language Recognition System

A real-time sign language recognition system that uses computer vision and deep learning to detect and classify hand gestures from camera input.

## Features

- **Real-time Detection**: Live camera feed with instant sign language recognition
- **Multiple Gestures**: Supports 6 common sign language gestures (hello, thanks, yes, no, iloveu, sad)
- **Hand Tracking**: Uses MediaPipe for accurate hand landmark detection
- **Deep Learning Model**: CNN-based classifier with transfer learning options
- **Data Visualization**: Training history plots and confusion matrix analysis
- **Easy Training**: Simple training pipeline with your own dataset

## Supported Gestures

The system currently recognizes the following sign language gestures:
- **Hello**: Greeting gesture
- **Thanks**: Thank you gesture  
- **Yes**: Affirmative gesture
- **No**: Negative gesture
- **I Love You**: Combined gesture
- **Sad**: Sadness expression

## Project Structure

```
signrecognation/
├── train/                 # Training data (images + XML annotations)
├── test/                  # Test data (images + XML annotations)
├── data_loader.py         # Data loading and preprocessing
├── model.py              # CNN model architecture
├── hand_detector.py      # MediaPipe hand detection
├── real_time_detection.py # Main real-time detection script
├── train_model.py        # Training script with evaluation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import cv2, mediapipe, tensorflow; print('All dependencies installed successfully!')"
   ```

## Usage

### 1. Training the Model

To train the model with your data:

```bash
python train_model.py
```

This will:
- Load your training and test data
- Show sample images from your dataset
- Let you choose between simple CNN or transfer learning
- Train the model with data augmentation
- Display training progress and evaluation metrics
- Save the trained model as `sign_language_model.h5`

### 2. Real-time Detection

To run real-time sign language detection:

```bash
python real_time_detection.py
```

Or if you want to train first and then run detection:

```bash
python real_time_detection.py --train
```

### 3. Testing Individual Components

Test hand detection:
```bash
python hand_detector.py
```

Test data loading:
```bash
python data_loader.py
```

## Data Format

The system expects your data to be organized as follows:

### Directory Structure
```
train/
├── gesture1.image1.jpg
├── gesture1.image1.xml
├── gesture1.image2.jpg
├── gesture1.image2.xml
└── ...

test/
├── gesture1.image1.jpg
├── gesture1.image1.xml
├── gesture1.image2.jpg
├── gesture1.image2.xml
└── ...
```

### XML Annotation Format
The XML files should be in Pascal VOC format:
```xml
<annotation>
    <filename>image.jpg</filename>
    <object>
        <name>gesture_name</name>
        <bndbox>
            <xmin>184</xmin>
            <ymin>147</ymin>
            <xmax>342</xmax>
            <ymax>350</ymax>
        </bndbox>
    </object>
</annotation>
```

## Model Architecture

The system provides two model options:

### 1. Simple CNN
- Lightweight architecture
- Faster training
- Good for smaller datasets
- 4 convolutional layers with max pooling
- Dropout for regularization

### 2. Transfer Learning (MobileNetV2)
- Pre-trained on ImageNet
- Better accuracy
- Slower training
- Requires more computational resources
- Fine-tuned for sign language recognition

## Performance Optimization

### For Better Accuracy:
1. **More Training Data**: Collect more images for each gesture
2. **Data Augmentation**: The system automatically applies rotation, scaling, and flipping
3. **Transfer Learning**: Use MobileNetV2 for better feature extraction
4. **Hyperparameter Tuning**: Adjust epochs, batch size, and learning rate

### For Real-time Performance:
1. **Lower Resolution**: Reduce camera resolution if needed
2. **Simple CNN**: Use the lightweight model for faster inference
3. **GPU Acceleration**: Use CUDA-enabled TensorFlow for faster training

## Troubleshooting

### Common Issues:

1. **Camera not working**:
   - Check if your camera is connected and not used by another application
   - Try different camera indices (0, 1, 2...)

2. **Model not loading**:
   - Make sure you've trained the model first using `train_model.py`
   - Check if `sign_language_model.h5` exists in the project directory

3. **Poor detection accuracy**:
   - Ensure good lighting conditions
   - Keep your hand clearly visible to the camera
   - Make sure your gestures match the training data
   - Retrain with more diverse data

4. **Memory issues**:
   - Reduce batch size during training
   - Use the simple CNN model instead of transfer learning
   - Close other applications to free up memory

### Dependencies Issues:

If you encounter dependency conflicts:
```bash
# Create a virtual environment
python -m venv signlang_env
source signlang_env/bin/activate  # On Windows: signlang_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Customization

### Adding New Gestures:
1. Add new gesture images and XML files to train/ and test/ directories
2. Update the `classes` list in `data_loader.py`
3. Retrain the model

### Modifying Model Architecture:
Edit the `model.py` file to customize the neural network architecture.

### Adjusting Detection Parameters:
Modify confidence thresholds and smoothing parameters in `real_time_detection.py`.

## Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving model accuracy
- Optimizing performance
- Adding support for more gestures
- Enhancing the user interface

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe for hand detection and landmark extraction
- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision operations
- The sign language community for gesture definitions

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are properly installed
4. Verify your data format matches the expected structure 