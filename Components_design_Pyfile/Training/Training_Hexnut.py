"""
AI Physics-Informed Hex Nut Parameter Enhancer
Neural Model With Pattern-Learning Verification
"""

import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# ======================================================================
# 1️⃣ Physics-Informed Synthetic Hex Nut Data Generator
# ======================================================================

def generate_hexnut_data(n_samples=60000, seed=42):

    np.random.seed(seed)
    random.seed(seed)

    materials = ["CarbonSteel", "Stainless", "Alloy"]
    coatings = ["BlackOxide", "Zinc", "Phosphate"]

    data = []

    for _ in range(n_samples):

        # INPUT PARAMETERS
        nominal_diameter = np.random.uniform(5, 48)
        thread_pitch = np.random.uniform(0.5, 6)

        nut_thickness = nominal_diameter * np.random.uniform(0.7, 1.1)
        width_across_flats = nominal_diameter * np.random.uniform(1.4, 1.8)

        chamfer = nominal_diameter * np.random.uniform(0.05, 0.12)
        hole_depth = nut_thickness * np.random.uniform(0.9, 1.1)

        material = random.choice(materials)
        coating = random.choice(coatings)

        # Physics/Manufacturing Factors
        mat_factor = {
            "CarbonSteel": 1.0,
            "Stainless": 0.92,
            "Alloy": 1.12
        }[material]

        coat_factor = {
            "BlackOxide": 1.00,
            "Zinc": 1.03,
            "Phosphate": 0.97
        }[coating]

        # OPTIMIZED OUTPUT PARAMETERS (TARGETS)
        nut_thickness_opt = nut_thickness * mat_factor * coat_factor
        width_flat_opt = width_across_flats * (1 + (thread_pitch / 50))
        chamfer_opt = chamfer * (1 + nominal_diameter / 200)
        hole_depth_opt = hole_depth * (1 + thread_pitch / 80)

        data.append([
            nominal_diameter, thread_pitch, nut_thickness,
            width_across_flats, chamfer, hole_depth,
            material, coating,
            nut_thickness_opt, width_flat_opt,
            chamfer_opt, hole_depth_opt
        ])

    cols = [
        "nominal_d", "pitch", "nut_thickness", "width_flats",
        "chamfer", "hole_depth", "material", "coating",

        # TARGETS
        "nut_thickness_opt", "width_flats_opt",
        "chamfer_opt", "hole_depth_opt"
    ]

    return pd.DataFrame(data, columns=cols)


# ======================================================================
# 2️⃣ Generate Dataset
# ======================================================================

print("🧩 Generating synthetic HEX-NUT dataset...")
df = generate_hexnut_data()
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape, "\n")


# ======================================================================
# 3️⃣ Feature / Target Split
# ======================================================================

X = df[[
    "nominal_d", "pitch", "nut_thickness", "width_flats",
    "chamfer", "hole_depth", "material", "coating"
]]

y = df[[
    "nut_thickness_opt", "width_flats_opt",
    "chamfer_opt", "hole_depth_opt"
]]


# ======================================================================
# 4️⃣ Preprocessing Setup
# ======================================================================

cat_features = ["material", "coating"]
num_features = [c for c in X.columns if c not in cat_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])


# ======================================================================
# 5️⃣ Train/Test Split
# ======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ======================================================================
# 6️⃣ Scale Target Variables
# ======================================================================

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


# ======================================================================
# 7️⃣ Transform Features
# ======================================================================

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)


# ======================================================================
# 8️⃣ Neural Network Model
# ======================================================================

model = Sequential([
    Dense(256, input_dim=X_train_proc.shape[1], activation="relu"),
    BatchNormalization(), Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(), Dropout(0.25),

    Dense(64, activation="relu"),

    Dense(y_train.shape[1], activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

history = model.fit(
    X_train_proc, y_train_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)


# ======================================================================
# 9️⃣ Evaluation
# ======================================================================

pred_scaled = model.predict(X_test_proc)
pred = scaler_y.inverse_transform(pred_scaled)

r2 = r2_score(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)

print("\n📊 Final HEX-NUT Model Evaluation:")
print(f"R² Score: {r2:.6f}")
print(f"RMSE: {rmse:.4f}\n")


# ======================================================================
# 🔟 Pattern-Learning Verification
# ======================================================================

print("🔍 Running pattern-learning verification...")

# A) Scatter Plots
cols = y.columns.tolist()
fig, axs = plt.subplots(1, 4, figsize=(20,4))

for i, col in enumerate(cols):
    axs[i].scatter(y_test[col], pred[:, i], s=5)
    axs[i].plot([y_test[col].min(), y_test[col].max()],
                [y_test[col].min(), y_test[col].max()], 'r')
    axs[i].set_title(col)

plt.show()

# B) Error Distribution
errors = y_test.values - pred
plt.hist(errors.flatten(), bins=40)
plt.title("Prediction Error Distribution")
plt.show()

# C) Physics Sensitivity Test
sample = X_test.iloc[5:6].copy()
values = []

for d in range(5, 60, 2):
    sample["nominal_d"] = d
    xproc = preprocessor.transform(sample)
    p = scaler_y.inverse_transform(model.predict(xproc))
    values.append([d, p[0][0]])

values = np.array(values)
plt.plot(values[:, 0], values[:, 1])
plt.title("Nominal Diameter → Nut Thickness Relationship")
plt.xlabel("nominal_d")
plt.ylabel("nut_thickness_opt")
plt.grid(True)
plt.show()


# ======================================================================
# 1️⃣1️⃣ Save Model + Scalers
# ======================================================================

BASE_DIR = r"C:\Users\Sanika\Documents\Project III\IEEE"
os.makedirs(BASE_DIR, exist_ok=True)

model.save(os.path.join(BASE_DIR, "hexnut_enhancer_model.h5"))
joblib.dump(preprocessor, os.path.join(BASE_DIR, "hexnut_preprocessor.pkl"))
joblib.dump(scaler_y, os.path.join(BASE_DIR, "hexnut_scaler_y.pkl"))

print("\n✅ HEX-NUT Model & Scalers Saved Successfully.")
