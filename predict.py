"""
Beverage Classification Prediction Script

This script loads a pre-trained MobileNetV2 model and predicts the class 
of a beverage image. It supports three classes: water, soda, and juice.

The model expects images to be preprocessed to 224x224 pixels with MobileNetV2
preprocessing (scaling to [-1, 1] range).

Usage:
    python predict.py <image_path>
    
Example:
    python predict.py ./data/test/water/water_test_001.jpg
    python predict.py my_beverage_photo.jpg

Output:
    Displays predicted class, confidence percentage, and probability 
    distribution across all classes with visual bars.

Requirements:
    - model.h5: Trained model file in the same directory
    - class_names.txt: Text file with class names (one per line)
    - TensorFlow/Keras installed
    
@author: Osmany
@version: 1.0
"""
import sys
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Relative paths
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

def load_class_names():
    """
    Load class names from class_names.txt file.
    
    Reads the class names file where each line contains one class name.
    The order of classes must match the model's output layer order.
    
    Returns:
        list: List of class names as strings (e.g., ['water', 'soda', 'juice'])
        
    Raises:
        FileNotFoundError: If class_names.txt doesn't exist
    """
    with open(CLASS_NAMES_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model prediction.
    
    Performs the following operations:
    1. Loads image from disk and resizes to target size
    2. Converts to numpy array
    3. Adds batch dimension
    4. Applies MobileNetV2 preprocessing (scales to [-1, 1])
    
    Args:
        img_path (str or Path): Path to the image file
        target_size (tuple): Target image size as (height, width). Default: (224, 224)
        
    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, 224, 224, 3)
                      ready for model prediction
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_path):
    """
    Predict the beverage class for a given image.
    
    Loads the trained model, preprocesses the input image, runs prediction,
    and displays results including predicted class, confidence, and probability
    distribution across all classes.
    
    Args:
        img_path (str or Path): Path to the image file to classify
        
    Returns:
        tuple: (predicted_class_name, confidence)
            - predicted_class_name (str): Name of predicted class
            - confidence (float): Confidence score between 0 and 1
            
    Example:
        >>> predict_image('water_bottle.jpg')
        📌 Predicted class: WATER
        📊 Confidence: 99.99%
    """
    # Load model and classes
    print(f"📂 Loading model from {MODEL_PATH}...")
    model = load_model(str(MODEL_PATH))
    class_names = load_class_names()
    
    print(f"✓ Classes: {class_names}\n")
    
    # Load and preprocess image
    print(f"🖼️  Processing image: {img_path}")
    img_array = load_and_preprocess_image(img_path)
    
    # Prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Results
    print("\n" + "=" * 60)
    print("🎯 PREDICTION RESULT")
    print("=" * 60)
    print(f"\n📌 Predicted class: {class_names[predicted_class_idx].upper()}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")
    
    print(f"\n📈 Probabilities by class:")
    for idx, class_name in enumerate(class_names):
        prob = predictions[0][idx] * 100
        bar = "█" * int(prob / 2)
        print(f"   {class_name:8s}: {prob:6.2f}% {bar}")
    
    return class_names[predicted_class_idx], confidence

def main():
    """
    Main entry point for the prediction script.
    
    Validates command-line arguments, checks if image file exists,
    and calls predict_image() function. Handles errors gracefully
    with informative messages.
    
    Command-line Args:
        image_path: Path to the image file to classify
        
    Exit Codes:
        0: Success
        1: Error (missing argument, file not found, or prediction failed)
    """
    if len(sys.argv) < 2:
        print("❌ Error: You must provide an image path")
        print("\nUsage: python predict.py <image_path>")
        print("Example: python predict.py test_image.jpg")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    # Verify it exists
    if not Path(img_path).exists():
        print(f"❌ Error: Image not found: {img_path}")
        sys.exit(1)
    
    # Predict
    try:
        predict_image(img_path)
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
