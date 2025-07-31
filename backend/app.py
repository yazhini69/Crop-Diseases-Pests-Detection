from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Adjust the model path as needed
model = load_model("plant_disease_model.h5")
CLASS_NAMES = [
    "Pepper_Bell_Bacterial_Spot",
    "Pepper_Bell_Healthy",
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight",
    "Tomato_Target_Spot",
    "Tomato_Mosaic_Virus",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites"
]

def predict_disease_from_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # match model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    # Defensive checks for model output
    if prediction is None or len(prediction) == 0:
        raise ValueError("Model returned empty prediction.")
    if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
        preds = prediction[0] if hasattr(prediction[0], "__len__") else prediction
    else:
        raise ValueError("Model output is not an array.")

    # Ensure output shape matches number of classes
    if len(preds) != len(CLASS_NAMES):
        raise ValueError(
            f"Model output shape mismatch. Expected {len(CLASS_NAMES)} classes, got {len(preds)}."
        )

    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]

    symptoms_treatment = {
        "Pepper_Bell_Bacterial_Spot": {
            "symptoms": "Dark, water-soaked spots on leaves and fruit.",
            "treatment": "Use copper-based bactericides and remove infected plants."
        },
        "Pepper_Bell_Healthy": {
            "symptoms": "No symptoms detected.",
            "treatment": "No treatment necessary."
        },
        "Potato_Early_Blight": {
            "symptoms": "Brown spots with concentric rings on leaves.",
            "treatment": "Apply fungicides and remove infected leaves."
        },
        "Potato_Healthy": {
            "symptoms": "No symptoms detected.",
            "treatment": "No treatment necessary."
        },
        "Potato_Late_Blight": {
            "symptoms": "Dark lesions on leaves and stems, rapid decay.",
            "treatment": "Destroy infected plants and apply fungicides."
        },
        "Tomato_Target_Spot": {
            "symptoms": "Small, brown circular lesions with concentric rings on leaves.",
            "treatment": "Remove affected leaves and use recommended fungicides."
        },
        "Tomato_Mosaic_Virus": {
            "symptoms": "Mottled, light and dark green areas on leaves and reduced fruit set.",
            "treatment": "Remove infected plants and control aphid vectors."
        },
        "Tomato_Yellow_Leaf_Curl_Virus": {
            "symptoms": "Yellowing and upward curling of leaves.",
            "treatment": "Control whitefly infestations and remove infected plants."
        },
        "Tomato_Bacterial_Spot": {
            "symptoms": "Small water-soaked spots on leaves, fruits, and stems.",
            "treatment": "Apply copper-based bactericides and ensure good air circulation."
        },
        "Tomato_Early_Blight": {
            "symptoms": "Brown spots with concentric rings, leaf yellowing.",
            "treatment": "Apply fungicides and remove infected leaves."
        },
        "Tomato_Healthy": {
            "symptoms": "No symptoms detected.",
            "treatment": "No treatment necessary."
        },
        "Tomato_Late_Blight": {
            "symptoms": "Dark, water-soaked lesions on leaves and fruit.",
            "treatment": "Use resistant varieties and apply fungicide."
        },
        "Tomato_Leaf_Mold": {
            "symptoms": "Yellow spots on upper leaf surface, mold on underside.",
            "treatment": "Increase airflow and apply fungicides."
        },
        "Tomato_Septoria_Leaf_Spot": {
            "symptoms": "Small circular spots with dark borders and gray centers.",
            "treatment": "Remove affected leaves and use fungicides."
        },
        "Tomato_Spider_Mites": {
            "symptoms": "Yellow stippling on leaves, webbing on foliage.",
            "treatment": "Use miticides or insecticidal soap and improve humidity."
        }
    }

    return {
        "predicted_class": predicted_class,
        **symptoms_treatment[predicted_class]
    }

@app.route('/')
def home():
    return "Crop Disease Detection API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed, please upload png/jpg/jpeg"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predict_disease_from_image(filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    if os.path.exists(filepath):
        os.remove(filepath)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
