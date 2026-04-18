
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "Placement Prediction API is running! Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        cgpa = float(data['cgpa'])
        iq = float(data['iq'])
        
        # Preprocess using the loaded scaler
        input_query = np.array([[cgpa, iq]])
        input_scaled = scaler.transform(input_query)
        
        # Predict
        result = model.predict(input_scaled)[0]
        
        return jsonify({'placement': int(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
