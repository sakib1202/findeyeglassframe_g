

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns



TRAIN_DATASET =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train face"
TEST_DATASET = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Test face"
TEST_FRAME = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train frame"

CATEGORIES=["heart","long","oval","round","square"]

import os
def validate_subfolders(path, categories):
    missing_folders = []
    for category in categories:
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            missing_folders.append(category)
    return missing_folders
TRAIN_DATASET =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train face"
TEST_DATASET = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Test face"
TEST_FRAME = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train frame"

CATEGORIES=["heart","long","oval","round","square"]

import os
def validate_subfolders(path, categories):
    missing_folders = []
    for category in categories:
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            missing_folders.append(category)
    return missing_folders

missing_folders_datatrain = validate_subfolders(path=TRAIN_DATASET, categories=CATEGORIES)
if missing_folders_datatrain:
    print("Missing subfolders in TRAIN_DATASET path:", missing_folders_datatrain)
else:
    print("All subfolders present in TRAIN_DATASET path.")

missing_folders_datatest = validate_subfolders(TEST_DATASET, CATEGORIES)
if missing_folders_datatest:
    print("Missing subfolders in TEST_DATASET path:", missing_folders_datatest)
else:
    print("All subfolders present in TEST_DATASET path.")

TRAIN_DATASET =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train face"
TEST_DATASET = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Test face"
TEST_FRAME = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train frame"

CATEGORIES=["heart","long","oval","round","square"]

import os
def validate_subfolders(path, categories):
    missing_folders = []
    for category in categories:
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            missing_folders.append(category)
    return missing_folders
TRAIN_DATASET =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train face"
TEST_DATASET = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Test face"
TEST_FRAME = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train frame"

CATEGORIES=["heart","long","oval","round","square"]

import os
def validate_subfolders(path, categories):
    missing_folders = []
    for category in categories:
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            missing_folders.append(category)
    return missing_folders

missing_folders_datatrain = validate_subfolders(path=TRAIN_DATASET, categories=CATEGORIES)
if missing_folders_datatrain:
    print("Missing subfolders in TRAIN_DATASET path:", missing_folders_datatrain)
else:
    print("All subfolders present in TRAIN_DATASET path.")

missing_folders_datatest = validate_subfolders(TEST_DATASET, CATEGORIES)
if missing_folders_datatest:
    print("Missing subfolders in TEST_DATASET path:", missing_folders_datatest)
else:
    print("All subfolders present in TEST_DATASET path.")
CATEGORIES=["heart","long","oval","round","square"]

missing_folders_datatest = validate_subfolders(TEST_DATASET, CATEGORIES)
if missing_folders_datatest:
    print("Missing subfolders in TEST_DATASET path:", missing_folders_datatest)
else:
    print("All subfolders present in TEST_DATASET path.")
CATEGORIES=["heart","long","oval","round","square"]

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train face"
test_dir = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Test face"
frame_dir =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train frame"

img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

import os
print("\n".join(os.listdir(r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set")))


train_dir =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train face"
test_dir = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Test face"
frame_dir =r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train frame"

if os.path.exists(train_dir):
    print(f"Training directory found: {train_dir}")
else:
    print(f"Training directory NOT found: {train_dir}")

     # Unzip again if extracted to an incorrect location

if os.path.exists(test_dir):
    print(f"Testing directory found: {test_dir}")
else:
    print(f"Testing directory NOT found: {test_dir}")

if os.path.exists(frame_dir):
    print(f"Training directory found: {frame_dir}")
else:
    print(f"Training directory NOT found: {frame_dir}")



train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

frame_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

frame_data = frame_datagen.flow_from_directory(
    frame_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

class_names = list(train_data.class_indices.keys())
print(f"Class Names: {class_names}")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import os

TRAIN_DATASET = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Train face"
TEST_DATASET = r"C:\Users\HP\sakib\findeyeglassframe_g-main\Face set\Test face"
CATEGORIES = ["heart", "long", "oval", "round", "square"]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    TRAIN_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    TRAIN_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=20, validation_data=validation_generator)

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

test_image_path = r"C:\Users\HP\Videos\New folder\tamim.jpg"
predicted_shape, frame_recommendation = recommend_glasses(test_image_path)
print(f"Predicted Face Shape: {predicted_shape}")
print(f"Recommended Glasses Frame: {frame_recommendation}")

import os
import matplotlib.pyplot as plt
import cv2 as cv


CATEGORIES = ["heart", "long", "oval", "round", "square"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

for i, category in enumerate(CATEGORIES):
    path = os.path.join(train_dir, category)
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

# Train the model
history = model.fit(train_generator, epochs=20, validation_data=validation_generator)

# Save the model
model.save('face_shape_model.h5')
