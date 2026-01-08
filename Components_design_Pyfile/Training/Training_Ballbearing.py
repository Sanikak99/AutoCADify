"""
AI Physics-Informed Bearing Parameter Enhancer
------------------------------------------------
Generates realistic synthetic data based on mechanical equations
and trains an ANN to predict optimized bearing geometry.
"""

import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# ===============================================================
# 1️⃣  Physics-informed Synthetic Data Generator
# ===============================================================
def generate_advanced_bearing_data(n_samples=5000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    data = []

    materials = ["Steel", "Ceramic", "Bronze"]
    lubes = ["Oil", "Grease", "Dry"]

    for _ in range(n_samples):
        # --- Base geometry ---
        d = np.random.uniform(10, 200)      # inner diameter
        B = np.random.uniform(0.25*d, 0.4*d)
        d_b = np.random.uniform(0.08*d, 0.15*d)
        C = np.random.uniform(0.001*d, 0.005*d)
        D = d + 2*(B/2 + d_b + C)
        z = int(np.pi * (D - d) / (1.15 * d_b))

        # --- Load & environmental conditions ---
        Fr = np.random.uniform(500, 5000)   # radial load
        Fa = np.random.uniform(0, 3000)     # axial load
        n = np.random.uniform(500, 10000)   # rpm
        mat = random.choice(materials)
        lub = random.choice(lubes)
        T = np.random.uniform(20, 120)      # temperature °C
        L10 = np.random.uniform(1e4, 1e6)   # life hours

        # --- Derived corrections ---
        mat_factor = {"Steel": 1.0, "Ceramic": 0.8, "Bronze": 1.2}[mat]
        load_factor = (Fr + 0.5*Fa) / 3000
        d_b_eff = d_b * (1 + 0.05*load_factor) * mat_factor
        z_eff = int(z * (1 + 0.1*load_factor))

        # Speed → width correction (high rpm → thinner)
        B_eff = B * (1 - min(n/20000, 0.3))

        # Temperature → clearance expansion
        C_eff = C * (1 + (T-20)/200)

        # Lubrication → width adjustment
        if lub == "Grease":
            B_eff *= 1.05
        elif lub == "Dry":
            B_eff *= 0.95

        D_eff = d + 2*(B_eff/2 + d_b_eff + C_eff)

        data.append([
            d, D_eff, B_eff, d_b_eff, C_eff, z_eff,
            Fr, Fa, n, mat, lub, T, L10
        ])

    cols = [
        "inner_d", "outer_d", "width", "ball_d", "clearance", "num_balls",
        "radial_load", "axial_load", "speed_rpm", "material",
        "lubrication", "temperature", "life_L10_hr"
    ]
    return pd.DataFrame(data, columns=cols)


# ===============================================================
# 2️⃣  Data Generation + Cleaning
# ===============================================================
print("🧩 Generating synthetic data...")
df = generate_advanced_bearing_data(60000)
print(f"Generated: {df.shape[0]} samples")

# Clean duplicates or unrealistic values
df.drop_duplicates(inplace=True)
df = df[(df["outer_d"] > df["inner_d"]) & (df["width"] > 0) & (df["ball_d"] > 0)]
df.reset_index(drop=True, inplace=True)
print(f"Cleaned data: {df.shape[0]} samples\n")

# ===============================================================
# 3️⃣  Feature / Target Split
# ===============================================================
# Features (input)
X = df[[
    "inner_d", "width", "ball_d", "num_balls",
    "radial_load", "axial_load", "speed_rpm",
    "material", "lubrication", "temperature", "life_L10_hr"
]]

# Targets (output — predicted optimized geometry)
y = df[["outer_d", "width", "ball_d", "num_balls"]]

# ===============================================================
# 4️⃣  Preprocessing Pipeline
# ===============================================================
cat_features = ["material", "lubrication"]
num_features = [c for c in X.columns if c not in cat_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# ===============================================================
# 5️⃣  Train / Test Split
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================================
# 6️⃣  Scale Targets Separately
# ===============================================================
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ===============================================================
# 7️⃣  Prepare Final Feature Matrices
# ===============================================================
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# ===============================================================
# 8️⃣  ANN Model
# ===============================================================
model = Sequential([
    Dense(256, input_dim=X_train_proc.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),

    Dense(y_train_scaled.shape[1], activation='linear')
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ===============================================================
# 9️⃣  Training
# ===============================================================
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train_proc, y_train_scaled,
    validation_split=0.2,
    epochs=120,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# ===============================================================
# 🔟  Evaluation
# ===============================================================
pred_scaled = model.predict(X_test_proc)
pred = scaler_y.inverse_transform(pred_scaled)
r2 = r2_score(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)

print(f"\n✅ Model Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}\n")

# ===============================================================
# 1️⃣1️⃣  Save Model + Scalers
# ===============================================================
base_dir = r"C:\Users\Sanika\Documents\Project III\IEEE"
os.makedirs(base_dir, exist_ok=True)

model.save(os.path.join(base_dir, "enhancer_model_physics.h5"))
joblib.dump(preprocessor, os.path.join(base_dir, "preprocessor.pkl"))
joblib.dump(scaler_y, os.path.join(base_dir, "scaler_y.pkl"))

print("✅ Model, preprocessor, and scaler saved successfully!\n")

# ===============================================================
# 1️⃣2️⃣  Plot Training Curve
# ===============================================================
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("ANN Training Progress (Physics-Informed Enhancer)")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
