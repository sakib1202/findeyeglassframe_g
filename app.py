from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__, template_folder="templates")  # Ensure Flask looks in 'templates/' for HTML files

# Define the path to the model
MODEL_PATH = r"C:\Users\HP\sakib\findeyeglassframe_g-main\face_shape_model.h5"

# Check if model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the pre-trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully from", MODEL_PATH)

# Ensure 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define class labels (Update these based on your dataset)
class_labels = {
    0: "Oval",
    1: "Round",
    2: "Square",
    3: "Heart",
    4: "Long"
}

# Define recommended frame shapes for each face shape
frame_recommendations = {
    "Oval": "Square, round, and aviator frames.",
    "Round": "Square or rectangular frames.",
    "Square": "Round or oval frames soften angular features.",
    "Heart": "Bottom-heavy frames or aviators balance a broader forehead.",
    "Long": "Cat-eye or oval frames complement cheekbones and soften angles."
}

@app.route('/')
def index():
    return render_template('eye.html')  # Ensure 'eye.html' is inside the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file temporarily
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        try:
            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)  # Preprocess for VGG16

            # Make the prediction
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction, axis=1)[0])  # Convert int64 to int
            predicted_label = class_labels.get(predicted_class, "Unknown")  # Get label from dictionary
            recommended_frame = frame_recommendations.get(predicted_label, "No recommendation available.")

            # Pass the prediction result to 'result.html'
            return render_template('result.html', 
                                   predicted_face_shape=predicted_label,
                                   recommended_frame_shape=recommended_frame)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
