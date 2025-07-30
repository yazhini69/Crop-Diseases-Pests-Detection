# 🌿 Plant Disease & Pest Detection 

This project provides a backend API that uses machine learning and computer vision to detect **plant diseases or pests** from uploaded images. Based on the diagnosis, the system returns actionable **treatment suggestions** to help farmers and agricultural professionals make informed decisions.

---

## 🚀 Features

- 📷 Upload an image of a plant (leaf, stem, etc.)
- 🧠 Get predicted disease or pest name
- 💡 Receive treatment suggestions and care tips
- ⚙️ Built with Flask and ready for integration with any frontend

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML
- **ML Model:** Pre-trained CNN (e.g., TensorFlow/Keras or PyTorch)
- **Image Processing** 
- **Deployment Ready:** Local server or cloud hosting (e.g., Render, Heroku)

---

## 📂 Project Structure
CropDiseaseDetection/
├── backend/
│ ├── app.py # Flask API backend
│ ├── model/ # ML model files (e.g., crop_recommendation_model.pkl)
│ ├── uploads/ # Uploaded images storage (used by backend)
│ ├── requirements.txt # Backend Python dependencies
│ └── venv/ # Python virtual environment folder (optional)
│
├── frontend/
│ ├── index.html # Frontend HTML with image upload & result display
│
├── raw/ # (Optional) Raw data or assets
└── README.md # Project documentation

---

## 🧪 API Endpoints

### 1. 📤 Upload & Predict Disease

**URL:** `/predict-image`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

#### ✅ Request:

Upload an image file using the `file` key.

curl -X POST http://localhost:5000/predict-image \
  -F "file=@path_to_your_image.jpg"
📬 Response:

{
  "predicted_class": "Powdery Mildew",
  "symptoms": "White powdery spots on leaves",
  "treatment": "Use fungicide spray and remove infected leaves"
}
🧰 Setup & Installation
1. Clone the Repository
git clone https://github.com/yourusername/plant-disease-api.git
cd plant-disease-api
2. Create Virtual Environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Server

python app.py
Server runs at: http://localhost:5001

🧠 Model Details
The model is trained on a dataset of annotated plant disease and pest images using a Convolutional Neural Network (CNN). You can replace the default pest_disease_model.pkl with your own trained model.

🔍 Example Images
Use the sample_images/ folder for quick testing. Upload them using Postman, curl, or your frontend.

✅ Future Improvements
🌐 Frontend UI with React or Flutter

🗣️ Multilingual treatment tips for farmers

📡 Cloud deployment with image upload tracking

🧪 More robust detection with larger datasets



📄 License
This project is licensed under the MIT License.


Open datasets from PlantVillage and Kaggle

OpenAI for inspiration and guidance


