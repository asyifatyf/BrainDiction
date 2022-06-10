import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image

model = keras.models.load_model("model.h5")

def trans_image(pillow_image):
    a = np.array(pillow_image.resize((128,128)))
    a = a.reshape(1,128,128,3)
    return a

def brain(predict):
    if predict==0:
        return 'Glioma Tumor'
    elif predict==1:
        return 'Meningioma Tumor'
    elif predict==2:
        return 'No Tumor or Normal'
    else:
        return 'Pituitary Tumor'
    
def predictions(x):
    a = model.predict(x)
    classi = np.where(a == np.amax(a))[1][0]
    return str(a[0][classi]*100), brain(classi)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method =="POST":
        file = request.files.get('file')
        if file is None:
            return jsonify({"message": "Error, no file"})
        else:
            try:
                pillow_img = Image.open(file)
                tensor = trans_image(pillow_img)
                percentage, prediction = predictions(tensor)
                data = {"prediction": prediction,
                       "percentage": percentage}
                return jsonify(data)
            except Exception as e:
                return jsonify({"error": str(e)})
    
    return "OK"

if __name__ == "__main__":
    app.run()