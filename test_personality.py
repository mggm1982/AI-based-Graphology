import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# =====================================
# LOAD TRAINED MODEL AND SCALER
# =====================================

# Replace these with your actual saved model/scaler file paths
MODEL_PATH = r"C:\Users\pk750\OneDrive\Desktop\data\models\personality_ann_model.h5"

SCALER_PATH = r"C:\Users\pk750\OneDrive\Desktop\data\models\scaler.pkl"
# Load the model
model = load_model(MODEL_PATH)

# Load the same scaler used during training
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# =====================================
# DEFINE NEW HANDWRITING SAMPLE FEATURES
# =====================================
# Change these values to test different handwriting profiles

Gender = 1          # 1 = Male, 0 = Female
Age = 25            # Example age
Writing_Speed_wpm = 42  # Example writing speed

# 15 handwriting features (values between 0 and 1)
features = np.array([
    0.7,  # Feature_1: Slant angle
    0.6,  # Feature_2: Letter spacing
    0.5,  # Feature_3: Pressure applied
    0.4,  # Feature_4: Line alignment
    0.8,  # Feature_5: Loop formation
    0.6,  # Feature_6: Stroke smoothness
    0.5,  # Feature_7: Word spacing variability
    0.7,  # Feature_8: Letter size consistency
    0.6,  # Feature_9: Pen lift frequency
    0.5,  # Feature_10: Legibility
    0.4,  # Feature_11: Cursive tendency
    0.6,  # Feature_12: Speed fluctuation
    0.7,  # Feature_13: Pen tilt direction
    0.5,  # Feature_14: Ink darkness
    0.6   # Feature_15: Number of strokes per letter
])

# Combine everything into one input vector
input_data = np.concatenate(([Writing_Speed_wpm, Gender, Age], features)).reshape(1, -1)

# Scale input (same way as training data)
input_scaled = scaler.transform(input_data)

# =====================================
# PREDICT PERSONALITY TRAITS
# =====================================
y_pred = model.predict(input_scaled)

traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

print("\n🧠 Predicted Personality Traits:")
for t, val in zip(traits, y_pred[0]):
    print(f"{t}: {val:.3f}")
