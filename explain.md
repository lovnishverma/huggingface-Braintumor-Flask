### **Code Explanation for Brain Tumor Detection Application Using Flask**

This Flask-based application allows users to upload an image (like an MRI scan) to predict if a brain tumor is detected using a pre-trained CNN (Convolutional Neural Network) model. The results, along with user details, are saved in a MongoDB database for review.

---

### **1. Importing Required Libraries**

```python
from flask import Flask, flash, request, redirect, render_template
import os
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import tempfile
from pymongo import MongoClient
from datetime import datetime
```

- **Flask**: A lightweight web framework to create web applications.
- **OpenCV (cv2)**: Library for image processing.
- **imutils**: Helper functions for image manipulation.
- **NumPy**: Array and mathematical operations.
- **TensorFlow/Keras**: To load the pre-trained brain tumor detection model.
- **MongoDB**: To store user inputs and model predictions.
- **Werkzeug**: For securely handling file uploads.
- **Datetime**: To save the timestamp for each prediction.

---

### **2. Loading the Pre-trained Model**

```python
braintumor_model = load_model('models/braintumor.h5')
```
- The brain tumor model (`braintumor.h5`) is loaded. This is a CNN-based model trained to detect brain tumors from images.

---

### **3. Flask Application Configuration**

```python
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.secret_key = "nielitchandigarhpunjabpolice"
```

- Flask is initialized, and caching for images is disabled to ensure updated images load after upload.
- A **secret key** is set for session management, which helps in managing messages (like flash messages).

---

### **4. MongoDB Connection**

```python
client = MongoClient("mongodb+srv://test:test@cluster0.sxci1.mongodb.net/?retryWrites=true&w=majority")
db = client['brain_tumor_detection']  # Database name
collection = db['predictions']  # Collection name
```
- Connects to **MongoDB Atlas** (cloud-hosted database).
- A database named `brain_tumor_detection` and collection `predictions` are created to store user details and predictions.

---

### **5. File Upload Helper Function**

```python
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
```
- Only files with extensions **png, jpg, jpeg** are allowed to ensure proper image input.

---

### **6. Image Preprocessing**

**a. `preprocess_imgs`: Resizing Images**

```python
def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(img)
    return np.array(set_new)
```
- Resizes the input image to a specific size (224x224) required by the model.

**b. `crop_imgs`: Region of Interest (ROI) Extraction**

```python
def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
        set_new.append(new_img)
    return np.array(set_new)
```
- This function identifies the **region of interest (ROI)**, cropping only the area where the brain is located for better accuracy.

---

### **7. Routes in Flask**

**a. `/` Route - Main Page**

```python
@app.route('/')
def brain_tumor():
    return render_template('braintumor.html')
```
- Displays the main upload form (`braintumor.html`).

**b. `/resultbt` Route - Prediction**

```python
@app.route('/resultbt', methods=['POST'])
def resultbt():
    # 1. Extract user inputs
    firstname = request.form['firstname']
    file = request.files['file']
    
    # 2. Validate image
    if file and allowed_file(file.filename):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # 3. Process Image
        img = cv2.imread(temp_file.name)
        img = crop_imgs([img])
        img = preprocess_imgs([img], (224, 224))

        # 4. Predict
        pred = braintumor_model.predict(img)
        prediction = 'Tumor Detected' if pred[0][0] >= 0.5 else 'No Tumor Detected'
        confidence_score = float(pred[0][0])

        # 5. Save to MongoDB
        result = {
            "firstname": firstname,
            "prediction": prediction,
            "confidence_score": confidence_score,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(result)

        # 6. Return Results
        return render_template('resultbt.html', r=prediction)
    else:
        flash('Invalid file format!')
        return redirect(request.url)
```
- **Step-by-step Flow**:
  1. Accept user inputs and uploaded image.
  2. Validate the file format.
  3. Preprocess the image (crop and resize).
  4. Use the CNN model to predict if there is a tumor.🚀
  5. Save the prediction and user details to MongoDB.
  6. Return the result.

**c. `/dbresults` Route - Fetch Predictions**

```python
@app.route('/dbresults')
def dbresults():
    all_results = collection.find().sort("timestamp", -1)
    tumor_count = sum(1 for r in all_results if r['prediction'] == 'Tumor Detected')
    total_patients = collection.count_documents({})
    return render_template('dbresults.html', total_patients=total_patients, tumor_count=tumor_count)
```
- Fetches all predictions from MongoDB and aggregates results (total patients, tumors detected).

---

### **8. Running the App**

```python
if __name__ == '__main__':
    app.run(debug=True)
```
- Runs the Flask application in **debug mode**.

---

### **Summary of Flow**
1. User uploads an MRI image and provides basic details.
2. Image is preprocessed (cropped, resized) for the CNN model.
3. The model predicts if a **brain tumor is detected** or not.
4. Results are stored in MongoDB and displayed back to the user.
5. Admins can view all results via the `/dbresults` route.

---

### **Tips**
1. **Flask Routes** handle user requests (`/`, `/resultbt`, `/dbresults`).
2. **OpenCV** helps preprocess images.
3. **MongoDB** stores user details and model predictions.
4. **Model Prediction** is done via a pre-trained Keras model.
5. Templates (`braintumor.html`, `resultbt.html`) are used to display data.

