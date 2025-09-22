from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

# Make sure models folder exists
os.makedirs('models', exist_ok=True)

# Dummy features (20 samples, 3 features each)
X = np.random.rand(20, 3)

# Dummy labels (emotions)
y = np.random.choice(["Calm", "Happy", "Stressed", "Tired"], 20)

# Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model in the models folder
joblib.dump(model, "models/emotion_model.pkl")
print("Dummy emotion model saved in models/emotion_model.pkl")