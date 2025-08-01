import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class labels (adjust if needed)
class_labels = ["pituitary", "glioma", "notumor", "meningioma"]

# Create app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model('model/model.h5')  # Adjust path if needed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(path):
    img = load_img(path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    predicted_label = class_labels[predicted_class_index]

    if predicted_label == "notumor":
        result = "No Tumor Detected"
    else:
        result = f"Tumor Detected: {predicted_label}"

    return result, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result, confidence = predict_image(file_path)
            image_url = file_path

    return render_template("index.html", result=result, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
