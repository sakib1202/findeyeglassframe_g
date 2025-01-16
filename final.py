# -*- coding: utf-8 -*-
"""final.ipynb"""

# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import load_img, img_to_array  # Updated imports for compatibility
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2 as cv

# Define dataset paths
TRAIN_DATASET = r"C:\Users\HP\Videos\New folder\Face set\Train face"
TEST_DATASET = r"C:\Users\HP\Videos\New folder\Face set\Test face"
FRAME_DATASET = r"C:\Users\HP\Videos\New folder\Face set\Train frame"

# Categories for face shapes
CATEGORIES = ['heart', 'long', 'oval', 'round', 'square']

# Validate subfolders
def validate_subfolders(path, categories):
    missing_folders = [category for category in categories if not os.path.exists(os.path.join(path, category))]
    return missing_folders

# Validate training dataset subfolders
missing_folders_datatrain = validate_subfolders(TRAIN_DATASET, CATEGORIES)
if missing_folders_datatrain:
    print(f"Missing subfolders in TRAIN_DATASET path: {missing_folders_datatrain}")
else:
    print("All required subfolders exist in the TRAIN_DATASET path.")

# Validate testing dataset subfolders
missing_folders_datatest = validate_subfolders(TEST_DATASET, CATEGORIES)
if missing_folders_datatest:
    print(f"Missing subfolders in TEST_DATASET path: {missing_folders_datatest}")
else:
    print("All required subfolders exist in the TEST_DATASET path.")

# Image data preparation
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Training and testing data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = train_datagen.flow_from_directory(
    TRAIN_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    TEST_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

frame_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

frame_data = frame_datagen.flow_from_directory(
    FRAME_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Display class names
class_names = list(train_data.class_indices.keys())
print(f"Class Names: {class_names}")

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(train_data, epochs=20, validation_data=test_data)

# Recommendations for glasses based on face shape
recommendations = {
    "heart": "Round or oval frames",
    "long": "Tall frames with decorative temples",
    "oval": "Square or rectangular frames",
    "round": "Square or angular frames",
    "square": "Round or oval frames"
}

# Function to recommend glasses
def recommend_glasses(image_path):
    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)  # Updated to use tf.keras.utils.load_img
    img_array = img_to_array(img) / 255.0  # Updated to use tf.keras.utils.img_to_array
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_shape = CATEGORIES[predicted_index]
    return predicted_shape, recommendations.get(predicted_shape, "No recommendation available")

# Example usage
test_image_path = r"C:\Users\HP\Videos\New folder\Face set\example.jpg"  # Update with an example image path
predicted_shape, frame_recommendation = recommend_glasses(test_image_path)
print(f"Predicted Face Shape: {predicted_shape}")
print(f"Recommended Glasses Frame: {frame_recommendation}")

# Visualization of categories
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, category in enumerate(CATEGORIES):
    path = os.path.join(TRAIN_DATASET, category)
    img_files = os.listdir(path)

    if img_files:
        img_file = img_files[0]
        img_path = os.path.join(path, img_file)

        img = cv.imread(img_path, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"Frame for {category} face shape")
        axes[i].axis('off')
    else:
        print(f"No images found in category: {category}")

for j in range(len(CATEGORIES), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
