import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

# Initialize Flask app
app = Flask(__name__)

# Path to the trained model (use relative path for deployment)
MODEL_PATH = 'model.h5'

# Categories for face shapes
CATEGORIES = ["heart", "long", "oval", "round", "square"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Recommendation logic for each face shape
recommendations = {
    "heart": "Round or oval frames",
    "long": "Tall frames with decorative temples",
    "oval": "Square or rectangular frames",
    "round": "Square or angular frames",
    "square": "Round or oval frames"
}

# Load the trained model
model = load_model(MODEL_PATH)

# Setup Image Data Generator
def create_data_generator(directory, is_training=False):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20 if is_training else 0,
        width_shift_range=0.2 if is_training else 0,
        height_shift_range=0.2 if is_training else 0,
        horizontal_flip=True if is_training else False
    )
    return datagen.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

# Paths for datasets
TRAIN_DATASET = 'static/Train face'
TEST_DATASET = 'static/Test face'
FRAME_DATASET = 'static/Train frame'

# Check if directories exist (relative to the Flask app)
for dir_name, directory in [("Training", TRAIN_DATASET), ("Testing", TEST_DATASET), ("Frame", FRAME_DATASET)]:
    if not os.path.exists(directory):
        print(f"{dir_name} directory NOT found: {directory}")

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded file temporarily
        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        # Predict the face shape
        predicted_shape, frame_recommendation = recommend_glasses(img_path)

        # Return the result as JSON
        return jsonify({
            "predicted_face_shape": predicted_shape,
            "recommended_glasses_frame": frame_recommendation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict face shape and recommend glasses
def recommend_glasses(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_shape = CATEGORIES[predicted_index]
    return predicted_shape, recommendations.get(predicted_shape, "No recommendation available")

# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
