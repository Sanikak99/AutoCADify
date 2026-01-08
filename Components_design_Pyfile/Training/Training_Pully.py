"""
AI Physics-Informed Pulley Parameter Enhancer
Training + Pattern Learning Verification
"""

import numpy as np
import pandas as pd
import random
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# ===============================================================
# 1️⃣ Physics-Informed Synthetic Pulley Data Generator
# ===============================================================

def generate_pulley_data(n_samples=60000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data = []

    for _ in range(n_samples):

        # Basic user inputs (features)
        outer_d = np.random.uniform(80, 600)
        bore_d = np.random.uniform(10, outer_d * 0.4)
        width = np.random.uniform(20, 150)

        groove_angle = np.random.uniform(28, 42)  # Standard V-groove angles
        side_width = np.random.uniform(3, 15)
        bottom_width = np.random.uniform(2, 10)

        # Physics-like transformation
        strength_factor = (outer_d / 100) * (width / 30)
        belt_tension = np.random.uniform(100, 2000)

        # Effective predictions (targets)
        outer_eff = outer_d * (1 + belt_tension / 300000)
        width_eff = width * (1 + strength_factor * 0.005)

        groove_depth = side_width * math.sin(math.radians(groove_angle/2))
        bottom_eff = bottom_width + groove_depth * 0.1
        angle_eff = groove_angle + np.random.uniform(-1.5, 1.5)

        data.append([
            outer_d, bore_d, width, groove_angle, side_width, bottom_width,
            outer_eff, width_eff, angle_eff, bottom_eff
        ])

    cols = [
        "outer_d", "bore_d", "width", "groove_angle",
        "groove_side_width", "bottom_groove_width",

        "outer_d_opt", "width_opt", "groove_angle_opt", "bottom_width_opt"
    ]

    return pd.DataFrame(data, columns=cols)


# ===============================================================
# 2️⃣ Generate Data
# ===============================================================

print("🧩 Generating pulley dataset...")
df = generate_pulley_data()
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape, "\n")


# ===============================================================
# 3️⃣ Feature / Target Split
# ===============================================================

X = df[[
    "outer_d", "bore_d", "width", "groove_angle",
    "groove_side_width", "bottom_groove_width"
]]

y = df[[
    "outer_d_opt", "width_opt", "groove_angle_opt", "bottom_width_opt"
]]


# ===============================================================
# 4️⃣ Scale Inputs & Outputs
# ===============================================================

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)


# ===============================================================
# 5️⃣ Train-Test Split
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)


# ===============================================================
# 6️⃣ Neural Network Model
# ===============================================================

model = Sequential([
    Dense(256, activation="relu", input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.25),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation="relu"),
    Dense(y_train.shape[1], activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=120,
    batch_size=32,
    callbacks=[es],
    verbose=1
)


# ===============================================================
# 7️⃣ Evaluate Model
# ===============================================================

pred_scaled = model.predict(X_test)
pred = scaler_y.inverse_transform(pred_scaled)

r2 = r2_score(y_test, pred_scaled)
rmse = mean_squared_error(y_test, pred_scaled, squared=False)

print("\n📊 Final Pulley Model Evaluation:")
print(f"R² Score: {r2:.6f}")
print(f"RMSE: {rmse:.4f}\n")


# ===============================================================
# 8️⃣ Pattern Verification (Optional Visualization)
# ===============================================================

cols = y.columns.tolist()
fig, axs = plt.subplots(1, 4, figsize=(18,4))
for i, col in enumerate(cols):
    axs[i].scatter(y_test[:, i], pred_scaled[:, i], s=5)
    axs[i].plot([min(y_test[:, i]), max(y_test[:, i])],
                [min(y_test[:, i]), max(y_test[:, i])], "r")
    axs[i].set_title(col)
plt.show()


# ===============================================================
# 9️⃣ Save Model
# ===============================================================

BASE = r"C:\Users\Sanika\Documents\Project III\IEEE"
os.makedirs(BASE, exist_ok=True)

model.save(os.path.join(BASE, "pulley_model.h5"))
joblib.dump(scaler_X, os.path.join(BASE, "pulley_scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(BASE, "pulley_scaler_y.pkl"))

print("✅ Pulley Model & Scalers Saved Successfully")
