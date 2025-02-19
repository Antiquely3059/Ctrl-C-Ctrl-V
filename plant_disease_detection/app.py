from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained simple CNN model (saved as .h5)
model = load_model('models/plant_disease_model.h5')

# Define the function to predict plant disease
def predict_plant_disease(img_path):
    # Load the image with the target size (150x150)
    img = image.load_img(img_path, target_size=(150, 150))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model's input shape (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image (scale pixel values to [0, 1])
    img_array = img_array / 255.0

    # Make the prediction
    prediction = model.predict(img_array)

    # Return prediction result: 1 is diseased, 0 is healthy
    return 'Diseased' if prediction[0] > 0.5 else 'Healthy'

# Route to render the index page (home page)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Save the uploaded file to the static/uploads folder
    uploads_folder = 'static/uploads/'
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    img_path = os.path.join(uploads_folder, file.filename)
    file.save(img_path)

    # Make the prediction using the trained model
    result = predict_plant_disease(img_path)

    # Return the result as a JSON response
    return jsonify({'result': result})

# Main entry point for the Flask application
if __name__ == '__main__':
    app.run(debug=True)
