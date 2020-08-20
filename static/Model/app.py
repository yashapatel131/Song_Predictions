from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import load_model


app = Flask(__name__)

@app.before_first_request
def load_model_to_app():
    app.predictor = load_model('./static/model/model.h5')

@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form['Acousticness'],
            request.form['Danceability'],
            request.form['Duration'], 
            request.form['Energy']
            request.form['Explicit'],
            request.form['Instrumentalness'],
            request.form['Liveness'], 
            request.form['Energy']]

    data = np.array([np.asarray(data, dtype=float)])

    predictions = app.predictor.predict(data)
    print('INFO Predictions: {}'.format(predictions))

    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]

    return render_template('index.html', pred=class_)