from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


MODEL_PATH = './models/model.h5'

model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    x = image.img_to_array(img)
    x=x/255
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    preds = model.predict(images ,batch_size = 5)
   
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        if preds[0]>0.5:
        	result = 'Rice Blast Disease'
        else:
        	result = 'Leaf Streak Disease'

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

