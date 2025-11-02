import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Load dataset
# ============================
data = pd.read_csv(r"C:\Users\pk750\OneDrive\Desktop\data\handwriting_dataset.csv")

# Drop ID-like column if present
if 'Handwriting_Sample' in data.columns:
    data.drop('Handwriting_Sample', axis=1, inplace=True)

# Encode categorical column (Gender)
if 'Gender' in data.columns:
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])  # Male=1, Female=0, Other=2

# ============================
# Define target columns (traits)
# ============================
target_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

X = data.drop(columns=target_cols)
y = data[target_cols]

# ============================
# Split and scale
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# Build ANN Model
# ============================
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(5, activation='linear')  # 5 outputs for personality traits
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])

# Early stopping for stable convergence
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ============================
# Train model
# ============================
history = model.fit(
    X_train_scaled, y_train,
    epochs=60,                     # 60 epochs for smoother accuracy curve
    batch_size=16,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

# ============================
# Evaluate
# ============================
y_pred = model.predict(X_test_scaled)

print("\n🧠 Model Evaluation Results:")
for i, trait in enumerate(target_cols):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    print(f"{trait}: R² = {r2:.3f}, MAE = {mae:.3f}")

# ============================
# Save Model and Predictions
# ============================
model.save(r"C:\Users\pk750\OneDrive\Desktop\data\models\personality_ann_model.h5")
print("\n✅ [INFO] Model saved successfully!")

results = pd.DataFrame({
    'Actual_Openness': y_test['Openness'],
    'Pred_Openness': y_pred[:, 0],
    'Actual_Conscientiousness': y_test['Conscientiousness'],
    'Pred_Conscientiousness': y_pred[:, 1],
    'Actual_Extraversion': y_test['Extraversion'],
    'Pred_Extraversion': y_pred[:, 2],
    'Actual_Agreeableness': y_test['Agreeableness'],
    'Pred_Agreeableness': y_pred[:, 3],
    'Actual_Neuroticism': y_test['Neuroticism'],
    'Pred_Neuroticism': y_pred[:, 4],
})

results.to_csv(r"C:\Users\pk750\OneDrive\Desktop\data\predicted_personality_results.csv", index=False)
print("✅ [INFO] Predictions saved successfully!")

# ============================
# Plot 1: Training vs Validation Loss
# ============================
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], 'b-', label='Training Loss')
plt.plot(history.history['val_loss'], 'r--', label='Validation Loss')
plt.title('Model Training Progress (Loss)', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(r"C:\Users\pk750\OneDrive\Desktop\data\models\loss_curve.png", dpi=300)
plt.show()

# ============================
# Plot 2: Training vs Validation Accuracy
# ============================
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], 'b-', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], 'orange', linestyle='--', label='Validation Accuracy')
plt.title('Training vs Testing Accuracy', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(r"C:\Users\pk750\OneDrive\Desktop\data\models\accuracy_curve.png", dpi=300)
plt.show()

# ============================
# Plot 3: Correlation Heatmap
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(pd.DataFrame(y_pred, columns=target_cols).corr(),
            annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Predicted Personality Trait Correlation", fontsize=13)
plt.tight_layout()
plt.savefig(r"C:\Users\pk750\OneDrive\Desktop\data\models\trait_correlation.png", dpi=300)
plt.show()

print("\n✅ All graphs saved in your 'models' folder successfully!")
