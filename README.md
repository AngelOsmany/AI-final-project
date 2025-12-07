# 🥤 Beverage Classifier with Transfer Learning

Deep Learning project using **Transfer Learning with MobileNetV2** to classify beverages into three categories: **water**, **soda**, and **orange juice**.

## 📋 Project Description

This classifier uses Transfer Learning to automatically identify beverage types in images. The model is based on **MobileNetV2** pre-trained on ImageNet with custom layers added for 3-class classification.

### 🎯 Model Performance
- **Test accuracy**: **96.92%**
- **Architecture**: MobileNetV2 + Transfer Learning
- **Classes**: water, soda, juice (orange juice)
- **Dataset**: 412 images (70% train / 15% val / 15% test)

### 🏗️ Model Architecture

```
MobileNetV2 (pre-trained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(3, activation='softmax')
```

**Parameters:**
- Total parameters: 2,422,339
- Trainable parameters: 164,355
- Input size: 224×224×3 (RGB)
- Batch size: 16
- Max epochs: 20
- Early stopping: patience=5
- Data augmentation: rotation, zoom, shifts, horizontal flip

## ⚙️ Requirements

### Python Installation

**Required:** Python 3.9

**Check your current version:**
```bash
python --version
```

**If you have a different version:**

1. Download Python 3.9 from: https://www.python.org/downloads/release/python-3918/
2. During installation, check "Add Python 3.9 to PATH"
3. Verify installation:
   ```bash
   python --version
   # Should show: Python 3.9.x
   ```

**If you have multiple Python versions:**
```bash
# Windows - Use py launcher
py -3.9 --version

# macOS/Linux - Use specific version
python3.9 --version
```

## 🚀 How to Use the Project

### Step 1: Download the Project

```bash
git clone https://github.com/AngelOsmany/AI-final-project.git
cd AI-final-project
```

### Step 2: Install Dependencies

**Windows:**
```powershell
# Create virtual environment with Python 3.9
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Create virtual environment with Python 3.9
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Make Predictions

The model is already trained and ready to use.

```bash
# Basic usage
python predict.py <path_to_image>

# Examples
python predict.py ./data/test/water/water_test_001.jpg
python predict.py my_photo.jpg
```

**Output example:**
```
🎯 PREDICTION RESULT
============================================================

📌 Predicted class: WATER
📊 Confidence: 99.99%

📈 Probabilities by class:
   water   :  99.99% █████████████████████████████████████████████████
   soda    :   0.00%
   juice   :   0.00%
```

### Step 4: Retrain the Model (Optional)

```bash
python train.py
```

**Training parameters:**
- Epochs: up to 20 (with early stopping)
- Batch size: 16
- Optimizer: Adam
- Loss: categorical_crossentropy
- Training time: 5-10 minutes (CPU) or 2-3 minutes (GPU)

**Output files:**
- `model.h5` - Trained model
- `class_names.txt` - Class labels
- Training plots displayed automatically

## 📊 Project Files

```
final/
├── data/                 # Dataset (412 images)
│   ├── train/           # 287 images (70%)
│   │   ├── water/      # 97 images
│   │   ├── soda/       # 102 images
│   │   └── juice/      # 88 images
│   ├── val/             # 60 images (15%)
│   └── test/            # 65 images (15%)
├── train.py             # Training script
├── predict.py           # Prediction script
├── model.h5             # Trained model (96.92% accuracy)
├── class_names.txt      # Class labels: water, soda, juice
└── requirements.txt     # Dependencies
```

## 📜 Credits

**Author**: AngelOsmany  
**Repository**: https://github.com/AngelOsmany/AI-final-project  
**Technologies**: TensorFlow, Keras, MobileNetV2, Python 3.9

---


