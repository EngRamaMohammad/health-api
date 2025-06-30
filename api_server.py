import os
import urllib.request

model_path = "health_risk_model.pkl"

if not os.path.exists(model_path):
    print("Downloading model file...")
  url = "https://drive.google.com/uc?export=download&id=1Y4aDtU0UPSeQmZr6J65h95jhIht6IxOo"

    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

model = joblib.load(model_path)



from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model once when the server starts
model = joblib.load("health_risk_model.pkl")

# Function to determine possible causes
def get_possible_causes(reading):
    heart_rate, temp, spo2, systolic_bp, diastolic_bp = reading
    reasons = []

    if heart_rate < 60:
        reasons.append("Bradycardia (slow heart rate)")
    elif heart_rate > 100:
        reasons.append("Tachycardia (fast heart rate)")

    if temp > 37.5:
        reasons.append("Fever (high temperature)")
    elif temp < 36:
        reasons.append("Hypothermia (low temperature)")

    if spo2 < 95:
        reasons.append("Hypoxia (low oxygen saturation)")

    if systolic_bp > 130 or diastolic_bp > 90:
        reasons.append("Hypertension (high blood pressure)")
    elif systolic_bp < 90 or diastolic_bp < 60:
        reasons.append("Hypotension (low blood pressure)")

    return reasons

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Extract data values in the expected order
    try:
        new_reading = [
            data['heart_rate'],
            data['temperature'],
            data['spo2'],
            data['systolic_bp'],
            data['diastolic_bp']
        ]
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {e}"}), 400

    # Make prediction
    prediction = model.predict([new_reading])[0]

    if prediction == 0:
        result = "Normal"
        causes = []
    else:
        result = "Abnormal"
        causes = get_possible_causes(new_reading)

    # Return the result as JSON
    return jsonify({
        "status": result,
        "possible_causes": causes
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
