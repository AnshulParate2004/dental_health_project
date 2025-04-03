from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Load the trained model with error handling
try:
    model = load_model(r'C:\Users\anshu\OneDrive\Documents\dental_health_project\caries_detection_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None  # Set to None if loading fails

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_caries(image_path):
    if model is None:
        return "Error: Model not loaded. Please check server logs."
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        if prediction > 0.5:
            return "Caries detected. Visit a dentist for a checkup."
        else:
            return "No caries detected. Maintain regular oral hygiene."
    except Exception as e:
        return f"Error processing image: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the image
        result = predict_caries(filepath)
        return jsonify({'message': 'File uploaded successfully', 'result': result})
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)