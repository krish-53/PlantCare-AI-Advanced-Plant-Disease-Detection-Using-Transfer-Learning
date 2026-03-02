from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model("models/plant_disease_model.h5")

class_names = os.listdir("dataset/train")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template("result.html",
                           prediction=predicted_class,
                           image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)