from flask import Flask, flash, request, redirect, render_template
import os
import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import tempfile
from pymongo import MongoClient
from datetime import datetime

# Load the Brain Tumor CNN Model
braintumor_model = load_model('models/braintumor_binary.h5')

# Configuring Flask application
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for images
app.secret_key = "nielitchandigarhpunjabpolice"  # Secret key for session management

# Allowed image file extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://test:test@cluster0.sxci1.mongodb.net/?retryWrites=true&w=majority")
db = client['brain_tumor_detection']  # Database name
collection = db['btpredictions']  # Collection name

def allowed_file(filename):
    """Check if the file is a valid image format (png, jpg, jpeg)."""
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def brain_tumor():
    """Render the HTML form for the user to upload an image."""
    return render_template('braintumor.html')

@app.route('/resultbt', methods=['POST'])
def resultbt():
    """Process the uploaded image and save prediction results to MongoDB."""
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']

        if file and allowed_file(file.filename):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            filename = secure_filename(file.filename)
            file.save(temp_file.name)

            flash('Image successfully uploaded and displayed below')

            try:
                # Load and preprocess the image
                img = load_img(temp_file.name, target_size=(128, 128))  # Resize image to match model's input size
                img_array = img_to_array(img)  # Convert image to array
                img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize image and add batch dimension

                # Make prediction
                pred = braintumor_model.predict(img_array)
                prediction = pred[0][0]
                confidence = prediction if prediction > 0.5 else 1 - prediction  # Calculate confidence
                predicted_class = 'Tumor Detected' if prediction > 0.5 else 'No Tumor Detected'  # Determine class based on threshold

                # Prepare data for MongoDB with JSON-serializable fields
                result = {
                    "firstname": firstname,
                    "lastname": lastname,
                    "email": email,
                    "phone": phone,
                    "gender": gender,
                    "age": age,
                    "image_name": filename,
                    "prediction": predicted_class,
                    "confidence_score": float(confidence),  # Ensure it's a standard float for JSON serialization
                    "timestamp": datetime.utcnow()
                }

                # Insert data into MongoDB
                collection.insert_one(result)

                # Return the result to the user
                return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=predicted_class, gender=gender)
            finally:
                os.remove(temp_file.name)  # Ensure temporary file is deleted
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

@app.route('/dbresults')
def dbresults():
    """Fetch all results from MongoDB, show aggregated data, and render in a template."""
    # Fetch all documents from MongoDB, sorted by timestamp in descending order
    all_results = collection.find().sort("timestamp", -1)  # Sort by timestamp, latest first

    # Convert cursor to a list of dictionaries
    results_list = []
    tumor_count = 0
    no_tumor_count = 0

    for result in all_results:
        result['_id'] = str(result['_id'])  # Convert ObjectId to string for JSON serialization
        results_list.append(result)

        # Count total patients with tumor and without tumor
        if result['prediction'] == 'Tumor Detected':
            tumor_count += 1
        else:
            no_tumor_count += 1

    total_patients = len(results_list)  # Total number of patients

    # Pass the results and aggregated counts to the HTML template
    return render_template('dbresults.html', 
                           results=results_list, 
                           total_patients=total_patients, 
                           tumor_count=tumor_count, 
                           no_tumor_count=no_tumor_count)


if __name__ == '__main__':
    app.run(debug=True)
