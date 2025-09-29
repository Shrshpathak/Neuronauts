import numpy as np, cv2, sounddevice as sd, librosa, time, traceback
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

fer_model = load_model(r"E:\internal_hackathon\models\video.h5")


audio_model = Sequential([
    Conv1D(256, 5, 1, 'same', activation='relu', input_shape=(162, 1)),
    MaxPooling1D(5, 2, 'same'),
    Conv1D(256, 5, 1, 'same', activation='relu'),
    MaxPooling1D(5, 2, 'same'),
    Conv1D(128, 5, 1, 'same', activation='relu'),
    MaxPooling1D(5, 2, 'same'),
    Dropout(0.2),
    Conv1D(64, 5, 1, 'same', activation='relu'),
    MaxPooling1D(5, 2, 'same'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')
])


audio_model.load_weights(r"E:\internal_hackathon\models\voice.h5")


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


last_emotion = None


input_mode = "emotion"


emotion_responses = {
    "Happy": "That's great to hear! Keep smiling and maybe share your joy with someone today.",
    "Sad": "I'm here for you. Try listening to some music or talking to a close friend — it might help.",
    "Fear": "It's okay to feel scared sometimes. Take a deep breath and remind yourself you're safe.",
    "Angry": "Take a moment to pause. A short walk or a glass of water might help you cool down.",
    "Surprise": "Wow, that sounds unexpected! Want to tell me more about what surprised you?",
    "Disgust": "That must have been unpleasant. Try shifting your focus to something positive.",
    "Neutral": "All calm and steady. Want to chat about something interesting?",
    "Uncertain": "I'm not sure how you're feeling. Want to talk about it or do something relaxing?"
}

manual_triggers = {
    "lonely": "You're doing amazing work — like an astronaut on a mission! Talk to me anytime, I'm here for you.",
    "tired": "Even heroes need rest. Take a break, breathe, and come back stronger.",
    "confused": "It's okay to feel lost sometimes. You're learning and growing — keep going!",
    "motivated": "That's the spirit! You're on the right path, keep pushing forward.",
    "bored": "Let’s explore something new together. What excites you these days?"
}

def set_input_mode(mode):
    global input_mode
    if mode in ["emotion", "manual"]:
        input_mode = mode
        print(f"🔁 Chatbot mode switched to: {input_mode}")
    else:
        print("❌ Invalid mode. Use 'emotion' or 'manual'.")

def get_chatbot_reply(emotion=None, user_input=None):
    if input_mode == "emotion" and emotion:
        return emotion_responses.get(emotion, "I'm here to chat whenever you're ready.")
    elif input_mode == "manual" and user_input:
        for keyword in manual_triggers:
            if keyword in user_input.lower():
                return manual_triggers[keyword]
        return f"You said: '{user_input}'. I'm here to talk more about it!"
    else:
        return "I'm not sure what to respond to right now."

def get_video_input(shared_frame):
    try:
        if shared_frame is None:
            print("No shared frame available")
            return None
        gray = cv2.cvtColor(shared_frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        return resized
    except Exception as e:
        print("Webcam error:", e)
        traceback.print_exc()
        return None

def get_audio_input(duration=3, sr=22050):
    try:
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        y = recording.flatten()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = mfcc.flatten()[:162]
        mfcc = np.pad(mfcc, (0, 162 - len(mfcc)), mode='constant')
        return mfcc.reshape(1, 162, 1)
    except Exception as e:
        print("Mic error:", e)
        traceback.print_exc()
        return None

def log_prediction(emotion, source, confidence):
    with open("emotion_log.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {emotion} ({source}) [{confidence:.2f}]\n")

def predict_emotion(shared_frame):
    global last_emotion
    video_input = get_video_input(shared_frame)
    audio_input = get_audio_input()
    timestamp = time.strftime('%H:%M:%S')

    if video_input is None and audio_input is None:
        return {"error": "Both webcam and mic failed"}

    try:
        if video_input is None:
            audio_probs = audio_model.predict(audio_input)[0][:7]
            top_idx = np.argmax(audio_probs)
            top_emotion = emotion_labels[top_idx]
            confidence = float(audio_probs[top_idx])
            if confidence < 0.35:
                top_emotion = "Uncertain"
            last_emotion = top_emotion
            log_prediction(top_emotion, "audio_only", confidence)
            chatbot_reply = get_chatbot_reply(emotion=top_emotion)
            return {
                "emotion": top_emotion,
                "confidence": confidence,
                "source": "audio_only",
                "timestamp": timestamp,
                "audio_probs": audio_probs.tolist(),
                "chatbot_reply": chatbot_reply
            }

        if audio_input is None:
            video_probs = fer_model.predict(video_input)[0]
            top_idx = np.argmax(video_probs)
            top_emotion = emotion_labels[top_idx]
            confidence = float(video_probs[top_idx])
            if confidence < 0.3:
                top_emotion = "Uncertain"
            last_emotion = top_emotion
            log_prediction(top_emotion, "video_only", confidence)
            chatbot_reply = get_chatbot_reply(emotion=top_emotion)
            return {
                "emotion": top_emotion,
                "confidence": confidence,
                "source": "video_only",
                "timestamp": timestamp,
                "video_probs": video_probs.tolist(),
                "chatbot_reply": chatbot_reply
            }

        video_probs = fer_model.predict(video_input)[0]
        audio_probs = audio_model.predict(audio_input)[0][:7]
        fused_probs = 0.6 * video_probs + 0.4 * audio_probs
        sorted_indices = np.argsort(fused_probs)[::-1]
        top_idx = sorted_indices[0]
        top_emotion = emotion_labels[top_idx]
        confidence = float(fused_probs[top_idx])

        if top_emotion == last_emotion and confidence < 0.6:
            top_idx = sorted_indices[1]
            top_emotion = emotion_labels[top_idx]
            confidence = float(fused_probs[top_idx])
            with open("emotion_log.txt", "a") as f:
                f.write(f"{timestamp} - Switched from {last_emotion} to {top_emotion} due to repetition\n")

        last_emotion = top_emotion

        if confidence < 0.2:
            top_emotion = "Happy"

        log_prediction(top_emotion, "fused", confidence)
        chatbot_reply = get_chatbot_reply(emotion=top_emotion)
        return {
            "emotion": top_emotion,
            "confidence": confidence,
            "source": "fused",
            "timestamp": timestamp,
            "video_probs": video_probs.tolist(),
            "audio_probs": audio_probs.tolist(),
            "fused_probs": fused_probs.tolist(),
            "chatbot_reply": chatbot_reply
        }

    except Exception as e:
        print("Fusion error:", e)
        traceback.print_exc()
        return {"error": f"Fusion failed: {str(e)}"}


if __name__ == "__main__":
    while True:
        mode = input("Choose mode ('emotion' or 'manual'): ").strip()
        set_input_mode(mode)

        if input_mode == "manual":
            user_text = input("You: ")
            reply = get_chatbot_reply(user_input=user_text)
            print("Chatbot:", reply)
        else:
            # Simulate shared_frame input from webcam
            shared_frame = cv2.imread("sample_face.jpg")  # Replace with actual frame capture
