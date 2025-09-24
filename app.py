from flask import Flask, render_template, request, jsonify, Response
import random, joblib, numpy as np, requests, os, cv2, time
from fusion import predict_emotion

app = Flask(__name__)

# ✅ Shared camera instance and frame buffer
stream_camera = cv2.VideoCapture(0)
latest_frame = None  # Global frame for prediction

def gen_frames():
    global latest_frame
    while True:
        success, frame = stream_camera.read()
        if not success:
            time.sleep(0.1)
            continue
        latest_frame = frame.copy()  # ✅ Save frame for fusion.py
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        time.sleep(0.05)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/emotions")
def emotions():
    return render_template("emotions.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

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

emotion_history = []

@app.route("/api/predict_emotion", methods=["POST"])
def predict_emotion_api():
    print("🔍 Predicting emotion...")
    result = predict_emotion(latest_frame)  # ✅ Pass shared frame
    if "error" in result:
        print("❌ Prediction error:", result["error"])
        return jsonify({
            "emotion": "⚠️ Error",
            "confidence": 0,
            "source": "none",
            "message": result["error"]
        }), 500

    timestamp = time.strftime("%H:%M:%S")
    result["timestamp"] = timestamp
    emotion_history.append({
        "time": timestamp,
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "source": result["source"]
    })

    print(f"✅ Emotion: {result['emotion']} | Source: {result['source']} | Confidence: {result['confidence']:.2f}")
    return jsonify(result)

@app.route("/api/emotion_history")
def emotion_history_api():
    return jsonify(emotion_history[-20:])

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

@app.route("/api/chat", methods=["POST"])
def chat():
    user_text = request.json.get("text", "")
    if not user_text:
        return jsonify({"reply": "⚠️ No input received."}), 400

    try:
        HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        HF_TOKEN = "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"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": user_text}
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            return jsonify({"reply": f"⚠️ API error: {response.text}"}), 500

        data = response.json()
        reply = data[0]["generated_text"] if isinstance(data, list) and data else data.get("generated_text", "⚠️ No response from model.")
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)