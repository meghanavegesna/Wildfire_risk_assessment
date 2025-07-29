import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
application = Flask(__name__)
app = application

# Load model and scaler from 'models' directory
ridge_model = pickle.load(open("models/ridge.pk1", "rb"))
standard_scaler = pickle.load(open("models/scaler.pk1", "rb"))


# Route for home/index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Getting values from form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Preprocess input
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Make prediction
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])
    
    else:
        return render_template('home.html')

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

 