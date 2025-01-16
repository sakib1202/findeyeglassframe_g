# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
import cv2 as cv
import matplotlib.pyplot as plt

# Define Paths
BASE_DIR = r"C:\Users\HP\Videos\New folder\Face set"
TRAIN_DATASET = os.path.join(BASE_DIR, "Train face")
TEST_DATASET = os.path.join(BASE_DIR, "Test face")
FRAME_DATASET = os.path.join(BASE_DIR, "Train frame")

CATEGORIES = ['heart', 'long', 'oval', 'round', 'square']
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Validate Directories
def validate_subfolders(path, categories):
    missing_folders = [category for category in categories if not os.path.exists(os.path.join(path, category))]
    return missing_folders

def check_and_create_directories():
    for dataset in [TRAIN_DATASET, TEST_DATASET, FRAME_DATASET]:
        if not os.path.exists(dataset):
            os.makedirs(dataset)
        missing_folders = validate_subfolders(dataset, CATEGORIES)
        for folder in missing_folders:
            os.makedirs(os.path.join(dataset, folder))

check_and_create_directories()

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model Definition
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

# Model Training
EPOCHS = 20
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Save the Model
model.save("face_shape_classifier.h5")

# Glasses Frame Recommendations
recommendations = {
    "heart": "Round or oval frames",
    "long": "Tall frames with decorative temples",
    "oval": "Square or rectangular frames",
    "round": "Square or angular frames",
    "square": "Round or oval frames"
}

def recommend_glasses(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_shape = CATEGORIES[predicted_index]
    return predicted_shape, recommendations.get(predicted_shape, "No recommendation available")

# Test Image
test_image_path = r"C:\path_to_test_image.jpg"  # Replace with the path to your test image
if os.path.exists(test_image_path):
    predicted_shape, frame_recommendation = recommend_glasses(test_image_path)
    print(f"Predicted Face Shape: {predicted_shape}")
    print(f"Recommended Glasses Frame: {frame_recommendation}")
else:
    print(f"Test image not found: {test_image_path}")

# Visualize Sample Images
def visualize_samples():
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

visualize_samples()
