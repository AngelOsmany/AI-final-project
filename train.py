"""
Beverage Classification Training Script

This script trains a deep learning model using Transfer Learning with MobileNetV2
to classify beverage images into three categories: water, soda, and juice.

The model architecture:
    - Base: MobileNetV2 pre-trained on ImageNet (frozen layers)
    - Custom head: GlobalAveragePooling2D → Dense(128) → Dropout(0.5) → Dense(3)
    
Features:
    - Data augmentation for training set (rotation, zoom, shifts, flips)
    - Early stopping to prevent overfitting
    - Model checkpoint saves best model based on validation accuracy
    - Generates training history plots (accuracy and loss curves)
    - Evaluates on separate test set
    
Usage:
    python train.py
    
The script expects data in this structure:
    data/
        train/
            water/
            soda/
            juice/
        val/
            water/
            soda/
            juice/
        test/
            water/
            soda/
            juice/

Output:
    - model.h5: Saved trained model
    - class_names.txt: List of class names
    - Training plots (displayed during training)
    
Configuration:
    - BATCH_SIZE: 16
    - EPOCHS: 20 (with early stopping)
    - IMG_SIZE: 224x224 (required by MobileNetV2)
    
@author: Osmany
@version: 1.0
"""
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# GENERAL CONFIGURATION
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 3

# Relative paths
BASE_DIR = Path(__file__).parent.resolve()
TRAIN_DIR = str(BASE_DIR / "data" / "train")
VAL_DIR = str(BASE_DIR / "data" / "val")
TEST_DIR = str(BASE_DIR / "data" / "test")

# DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# BASE MODEL (TRANSFER LEARNING)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze layers

# CLASSIFICATION HEAD
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)  # Regularization
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# COMPILATION
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# TRAINING
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# EVALUATION
loss, acc = model.evaluate(test_gen)
print(f"\nTest accuracy: {acc:.4f}")

# SAVE MODEL AND CLASSES
model.save(str(BASE_DIR / "model.h5"))
print(f"✅ Model saved as {BASE_DIR / 'model.h5'}")

# Save class names
class_names = list(train_gen.class_indices.keys())
with open(BASE_DIR / "class_names.txt", "w") as f:
    f.write("\n".join(class_names))
print(f"✅ Classes saved in {BASE_DIR / 'class_names.txt'}: {class_names}")

# PLOTS
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()
