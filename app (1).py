
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)

# Helper to load files
def load_file(name):
    try:
        return pickle.load(open(name, 'rb'))
    except Exception as e:
        print(f'Error loading {name}: {e}')
        return None

model = load_file('model.pkl')
scaler = load_file('scaler.pkl')

@app.route('/')
def home():
    status = 'Ready' if model and scaler else 'Error: Model/Scaler not loaded'
    return f'Placement Prediction API is {status}.'

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or Scaler not loaded on server'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        cgpa = float(data.get('cgpa', 0))
        iq = float(data.get('iq', 0))

        # Preprocess
        input_query = np.array([[cgpa, iq]])
        input_scaled = scaler.transform(input_query)

        # Predict
        result = model.predict(input_scaled)[0]

        return jsonify({'placement': int(result)})
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
