import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.optimizers import Adam

# Directory paths for training, testing, and frames
TRAIN_DATASET = r"C:\Users\HP\Videos\New folder\Face set\Train face"
TEST_DATASET = r"C:\Users\HP\Videos\New folder\Face set\Test face"
TEST_FRAME = r"C:\Users\HP\Videos\New folder\Face set\Train frame"

# Categories for face shapes
CATEGORIES = ["heart", "long", "oval", "round", "square"]

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Validate subfolders
def validate_subfolders(path, categories):
    """
    Validate that all category subfolders exist in the given path.

    Args:
        path (str): The main dataset folder path.
        categories (list): List of required category names.

    Returns:
        list: Missing subfolders.
    """
    missing_folders = [
        category for category in categories if not os.path.exists(os.path.join(path, category))
    ]
    return missing_folders

# Check training and test datasets for subfolder presence
for dataset, dataset_name in [(TRAIN_DATASET, "TRAIN_DATASET"), (TEST_DATASET, "TEST_DATASET")]:
    missing_folders = validate_subfolders(dataset, CATEGORIES)
    if missing_folders:
        print(f"Missing subfolders in {dataset_name}: {missing_folders}")
    else:
        print(f"All subfolders are present in {dataset_name}.")

# Check if directories exist
train_dir = TRAIN_DATASET
test_dir = TEST_DATASET
frame_dir = TEST_FRAME

# Confirm directories are valid
for dir_name, directory in [("Training", train_dir), ("Testing", test_dir), ("Frame", frame_dir)]:
    if os.path.exists(directory):
        print(f"{dir_name} directory found: {directory}")
    else:
        print(f"{dir_name} directory NOT found: {directory}")

# Proceed with data generators if directories are valid
if os.path.exists(train_dir) and os.path.exists(test_dir) and os.path.exists(frame_dir):
    # Data augmentation and preprocessing for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    # Data augmentation and preprocessing for testing
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Create training data generator
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Create testing data generator
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Create frame data generator
    frame_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    frame_data = frame_datagen.flow_from_directory(
        frame_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    class_names = list(train_data.class_indices.keys())
    print(f"Class Names: {class_names}")

    # Building the CNN model
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
        Dense(len(train_data.class_indices), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_data, epochs=20, validation_data=test_data)

    # Glasses frame recommendation based on predicted face shape
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

    # Test recommendation on an example image
    test_image_path = r"C:\Users\HP\Videos\New folder\tamim.jpg"
    predicted_shape, frame_recommendation = recommend_glasses(test_image_path)
    print(f"Predicted Face Shape: {predicted_shape}")
    print(f"Recommended Glasses Frame: {frame_recommendation}")

    # Displaying the frames for each face shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, category in enumerate(CATEGORIES):
        path = os.path.join(TEST_FRAME, category)
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

else:
    print("One or more directories are missing. Aborting model setup.")
