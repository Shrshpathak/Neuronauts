from flask import Flask, render_template, request, jsonify
import random
import joblib
import numpy as np
import requests
import os

# Load your ML model
emotion_model = joblib.load('models/emotion_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# New routes for Emotions and Symptoms
@app.route("/emotions")
def emotions():
    return render_template("emotions.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")


# API: Simulated vitals
@app.route('/api/vitals')
def vitals():
    heartbeat = random.randint(60, 110)
    oxygen = random.randint(90, 100)
    bp_systolic = random.randint(110, 130)
    bp_diastolic = random.randint(70, 85)
    temp = round(36 + random.random()*2, 1)
    return jsonify({
        'heartbeat': heartbeat,
        'oxygen': oxygen,
        'bp': f"{bp_systolic}/{bp_diastolic}",
        'temp': temp
    })


# API: Emotion detection
@app.route("/api/predict_emotion", methods=["POST"])
def predict_emotion():
    try:
        data = request.get_json()
        features = data.get("features", [])
        prediction = emotion_model.predict([features])[0]
        return jsonify({"emotion": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# API: Symptom Checker
@app.route('/api/symptom', methods=['POST'])
def symptom_checker():
    symptom_text = request.json.get('symptom', '').lower()
    result = "No condition detected."
    if "headache" in symptom_text and "nausea" in symptom_text:
        result = "Possible Space Motion Sickness."
    elif "fatigue" in symptom_text:
        result = "Possible Muscle Weakness."
    elif "cough" in symptom_text:
        result = "Possible Respiratory Issue."
    return jsonify({'result': result})


# ✅ API: Chatbot (Hugging Face)
@app.route("/api/chat", methods=["POST"])
def chat():
    user_text = request.json.get("text", "")
    if not user_text:
        return jsonify({"reply": "⚠️ No input received."}), 400

    try:
        HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        HF_TOKEN = "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"  # replace with env var in production

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": user_text}

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            return jsonify({"reply": f"⚠️ API error: {response.text}"}), 500

        data = response.json()

        reply = "⚠️ No response from model."
        if isinstance(data, list) and data and "generated_text" in data[0]:
            reply = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            reply = data["generated_text"]

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
