from flask import Flask, request, jsonify
from flask_cors import CORS  
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
# Folder to save uploaded images temporarily
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Dummy disease prediction function
def predict_disease_from_image(image_path):
    # TODO: Replace this with your actual model prediction code
    # e.g., load image, preprocess, model.predict(), etc.
    return {
        "predicted_class": "Powdery Mildew",
        "symptoms": "White powdery spots on leaves",
        "treatment": "Use fungicide spray and remove infected leaves"
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

    # Save file securely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Run your prediction logic here
        result = predict_disease_from_image(filepath)
    except Exception as e:
        os.remove(filepath)  # Clean up before returning error
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Clean up uploaded file
    os.remove(filepath)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
