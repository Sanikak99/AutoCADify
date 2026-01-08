"""
AI Physics-Informed Flange Parameter Enhancer
Pattern-Learning Verification Version
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
# 1️⃣ Physics-Informed Synthetic Flange Data Generator
# ======================================================================

def generate_flange_data(n_samples=60000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    materials = ["CarbonSteel", "Stainless", "Alloy", "CastIron"]
    gasket_types = ["SpiralWound", "RingJoint", "FlatMetal"]

    data = []

    for _ in range(n_samples):

        inner_d = np.random.uniform(20, 300)
        pressure_class = random.choice([150, 300, 600, 900])

        outer_d = inner_d * np.random.uniform(1.8, 2.5)
        thickness = (pressure_class / 150) * np.random.uniform(18, 38)

        bolt_circle_d = outer_d - np.random.uniform(20, 45)
        bolt_diameter = np.random.uniform(12, 28)
        num_bolts = random.choice([4, 8, 12, 16, 20, 24])

        gasket = random.choice(gasket_types)
        material = random.choice(materials)

        internal_pressure = np.random.uniform(2, 100)
        temperature = np.random.uniform(20, 450)
        axial_force = np.random.uniform(5e3, 2e5)
        bending_moment = np.random.uniform(100, 3000)

        mat_factor = {
            "CarbonSteel": 1.0,
            "Stainless": 0.9,
            "Alloy": 1.15,
            "CastIron": 0.75
        }[material]

        thickness_eff = thickness * (1 + internal_pressure / 500)
        outer_eff = outer_d * (1 + (temperature - 20) * 1e-4)
        num_bolts_eff = int(num_bolts * (1 + bending_moment / 30000))
        num_bolts_eff = max(4, min(32, num_bolts_eff))

        gasket_factor = {
            "SpiralWound": 1.0,
            "RingJoint": 1.1,
            "FlatMetal": 0.9
        }[gasket]

        bolt_diameter_eff = bolt_diameter * gasket_factor * mat_factor
        bolt_circle_eff = bolt_circle_d * (1 + axial_force / 1e6)

        data.append([
            inner_d, outer_d, thickness, bolt_circle_d, bolt_diameter,
            num_bolts, material, gasket,
            internal_pressure, temperature, axial_force, bending_moment,

            outer_eff, thickness_eff, bolt_circle_eff, bolt_diameter_eff, num_bolts_eff
        ])

    cols = [
        "inner_d", "outer_d", "thickness", "bolt_circle_d", "bolt_diameter",
        "num_bolts", "material", "gasket",
        "internal_pressure", "temperature", "axial_force", "bending_moment",

        "outer_d_opt", "thickness_opt", "bolt_circle_opt",
        "bolt_diameter_opt", "num_bolts_opt"
    ]

    return pd.DataFrame(data, columns=cols)



# ======================================================================
# 2️⃣ Generate Data
# ======================================================================

print("🧩 Generating dataset...")
df = generate_flange_data()
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape, "\n")



# ======================================================================
# 3️⃣ Split Features / Targets
# ======================================================================

X = df[[
    "inner_d", "outer_d", "thickness", "bolt_circle_d", "bolt_diameter",
    "num_bolts", "material", "gasket",
    "internal_pressure", "temperature", "axial_force", "bending_moment"
]]

y = df[[
    "outer_d_opt", "thickness_opt", "bolt_circle_opt",
    "bolt_diameter_opt", "num_bolts_opt"
]]



# ======================================================================
# 4️⃣ Preprocessing
# ======================================================================

cat_features = ["material", "gasket"]
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
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation="relu"),

    Dense(y_train_scaled.shape[1], activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train_proc, y_train_scaled,
    validation_split=0.2,
    epochs=120,
    batch_size=32,
    callbacks=[es],
    verbose=1
)



# ======================================================================
# 9️⃣ Evaluate Model
# ======================================================================

pred_scaled = model.predict(X_test_proc)
pred = scaler_y.inverse_transform(pred_scaled)

r2 = r2_score(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)

print("\n📊 Final Model Evaluation:")
print(f"R² Score: {r2:.6f}")
print(f"RMSE: {rmse:.4f}\n")



# ======================================================================
# 🔟 PATTERN-LEARNING VERIFICATION TESTS
# ======================================================================

print("🔍 Running Pattern-Learning Tests...\n")

# ------------------------------------------------------
# A) TRUE vs PREDICTED SCATTER PLOTS
# ------------------------------------------------------

cols = y.columns.tolist()

fig, axs = plt.subplots(1, 5, figsize=(22,4))
for i, col in enumerate(cols):
    axs[i].scatter(y_test[col], pred[:, i], s=5)
    axs[i].plot([y_test[col].min(), y_test[col].max()],
                [y_test[col].min(), y_test[col].max()], "r")
    axs[i].set_title(col)
plt.show()



# ------------------------------------------------------
# B) Error Distribution Plot
# ------------------------------------------------------

errors = y_test.values - pred
plt.hist(errors.flatten(), bins=40)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()



# ------------------------------------------------------
# C) Physics Sensitivity Test
# ------------------------------------------------------

sample = X_test.iloc[10:11].copy()
values = []

for d in range(50, 300, 10):
    sample["inner_d"] = d
    xproc = preprocessor.transform(sample)
    p = scaler_y.inverse_transform(model.predict(xproc))
    values.append([d, p[0][0]])

values = np.array(values)

plt.plot(values[:,0], values[:,1])
plt.xlabel("inner_d")
plt.ylabel("predicted outer_d_opt")
plt.title("Smoothness & Physics Consistency Test")
plt.grid(True)
plt.show()



# ------------------------------------------------------
# D) Feature Removal Stress Test
# ------------------------------------------------------

print("\n⚠️ Running stress-test: remove 'internal_pressure'...")

X_removed = X.copy()
X_removed["internal_pressure"] = 0

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_removed, y, test_size=0.2, random_state=42
)

X_train2_proc = preprocessor.transform(X_train2)
X_test2_proc = preprocessor.transform(X_test2)

pred_removed = model.predict(X_test2_proc)
pred_removed = scaler_y.inverse_transform(pred_removed)

r2_removed = r2_score(y_test2, pred_removed)

print(f"R² after removing key physics feature = {r2_removed:.4f} (Should Drop)")



# ======================================================================
# 1️⃣1️⃣ Save Model + Preprocessor + Scaler
# ======================================================================

BASE_DIR = r"C:\Users\Sanika\Documents\Project III\IEEE"
os.makedirs(BASE_DIR, exist_ok=True)

model.save(os.path.join(BASE_DIR, "flange_enhancer_model.h5"))
joblib.dump(preprocessor, os.path.join(BASE_DIR, "flange_preprocessor.pkl"))
joblib.dump(scaler_y, os.path.join(BASE_DIR, "flange_scaler_y.pkl"))

print("\n✅ Model & Scalers Saved.")
