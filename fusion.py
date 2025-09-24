import numpy as np, cv2, sounddevice as sd, librosa, time, traceback
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# Load FER model (video)
fer_model = load_model(r"E:\internal_hackathon\models\video.h5")

# Rebuild audio model architecture
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

# Load weights only
audio_model.load_weights(r"E:\internal_hackathon\models\voice.h5")

# Emotion labels (7 classes only)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ✅ Short-term memory to avoid repetition
last_emotion = None

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
        # Only audio available
        if video_input is None:
            audio_probs = audio_model.predict(audio_input)[0][:7]
            print("Audio probs:", audio_probs)
            top_idx = np.argmax(audio_probs)
            top_emotion = emotion_labels[top_idx]
            confidence = float(audio_probs[top_idx])
            if confidence < 0.35:
                top_emotion = "Uncertain"
            last_emotion = top_emotion
            log_prediction(top_emotion, "audio_only", confidence)
            return {
                "emotion": top_emotion,
                "confidence": confidence,
                "source": "audio_only",
                "timestamp": timestamp,
                "audio_probs": audio_probs.tolist()
            }

        # Only video available
        if audio_input is None:
            video_probs = fer_model.predict(video_input)[0]
            print("Video probs:", video_probs)
            top_idx = np.argmax(video_probs)
            top_emotion = emotion_labels[top_idx]
            confidence = float(video_probs[top_idx])
            if confidence < 0.5:
                top_emotion = "Uncertain"
            last_emotion = top_emotion
            log_prediction(top_emotion, "video_only", confidence)
            return {
                "emotion": top_emotion,
                "confidence": confidence,
                "source": "video_only",
                "timestamp": timestamp,
                "video_probs": video_probs.tolist()
            }

        # Both available — fusion
        video_probs = fer_model.predict(video_input)[0]
        audio_probs = audio_model.predict(audio_input)[0][:7]
        print("Video probs:", video_probs)
        print("Audio probs:", audio_probs)
        fused_probs = 0.6 * video_probs + 0.4 * audio_probs
        print("Fused probs:", fused_probs)

        sorted_indices = np.argsort(fused_probs)[::-1]
        top_idx = sorted_indices[0]
        top_emotion = emotion_labels[top_idx]
        confidence = float(fused_probs[top_idx])

        # ✅ Repetition break logic
        if top_emotion == last_emotion and confidence < 0.6:
            print(f"⚠️ Repeated emotion '{top_emotion}' detected. Switching to next best.")
            top_idx = sorted_indices[1]
            top_emotion = emotion_labels[top_idx]
            confidence = float(fused_probs[top_idx])
            with open("emotion_log.txt", "a") as f:
                f.write(f"{timestamp} - Switched from {last_emotion} to {top_emotion} due to repetition\n")

        last_emotion = top_emotion

        if confidence < 0.2:
            top_emotion = "Happy"

        log_prediction(top_emotion, "fused", confidence)
        return {
            "emotion": top_emotion,
            "confidence": confidence,
            "source": "fused",
            "timestamp": timestamp,
            "video_probs": video_probs.tolist(),
            "audio_probs": audio_probs.tolist(),
            "fused_probs": fused_probs.tolist()
        }

    except Exception as e:
        print("Fusion error:", e)
        traceback.print_exc()
        return {"error": f"Fusion failed: {str(e)}"}