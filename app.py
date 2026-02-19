# HematoVision - Flask Web App for Blood Cell Classification
# Student: Sivanandini Nallamekala
# Internship: SmartBridge (APSCHE VIP Program)

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask App
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/blood_cell.h5')

# Class Labels (Dataset Classes)
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# Home Page Route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Create static folder if not exists
    if not os.path.exists('static'):
        os.makedirs('static')

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]

    return render_template('result.html', prediction=predicted_class)

# Run the Application
if __name__ == '__main__':
    app.run(debug=True)
