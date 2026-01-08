"""
AI Physics-Informed Shaft Parameter Enhancer
Training File for Shaft Design Optimization
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
# 1️⃣ Physics-Informed Synthetic Shaft Data Generator
# ======================================================================

def generate_shaft_data(n_samples=60000, seed=42):
    """
    Generate synthetic shaft design data based on mechanical engineering principles
    """
    np.random.seed(seed)
    random.seed(seed)

    # Material properties
    materials = ["CarbonSteel", "Stainless304", "Alloy4140", "CastIron"]
    
    # Application types
    applications = ["Transmission", "PowerTransmission", "Spindle", "Axle"]

    data = []

    for _ in range(n_samples):
        # Input parameters
        length = np.random.uniform(50, 1000)  # mm
        diameter = np.random.uniform(20, 200)  # mm
        hollow_ratio = np.random.choice([0, 0.3, 0.5, 0.6, 0.7])  # 0 = solid shaft
        
        material = random.choice(materials)
        application = random.choice(applications)
        
        # Loading conditions
        torque = np.random.uniform(100, 50000)  # N·m
        bending_moment = np.random.uniform(50, 10000)  # N·m
        axial_force = np.random.uniform(0, 100000)  # N
        rpm = np.random.uniform(100, 5000)  # revolutions per minute
        temperature = np.random.uniform(20, 300)  # °C
        
        # Material properties mapping
        mat_properties = {
            "CarbonSteel": {"yield": 250, "modulus": 200, "density": 7850, "thermal_exp": 1.2e-5},
            "Stainless304": {"yield": 205, "modulus": 193, "density": 8000, "thermal_exp": 1.7e-5},
            "Alloy4140": {"yield": 415, "modulus": 205, "density": 7850, "thermal_exp": 1.1e-5},
            "CastIron": {"yield": 180, "modulus": 100, "density": 7200, "thermal_exp": 1.0e-5}
        }[material]
        
        yield_strength = mat_properties["yield"]  # MPa
        elastic_modulus = mat_properties["modulus"]  # GPa
        density = mat_properties["density"]  # kg/m³
        thermal_expansion = mat_properties["thermal_exp"]
        
        # ============================================================
        # PHYSICS-BASED OPTIMIZATION CALCULATIONS
        # ============================================================
        
        # 1. Torsional shear stress: τ = (T·r)/J
        # For solid shaft: J = π·d⁴/32
        # For hollow shaft: J = π(do⁴ - di⁴)/32
        inner_diameter = diameter * hollow_ratio
        
        if hollow_ratio == 0:
            polar_moment = (np.pi * diameter**4) / 32
        else:
            polar_moment = (np.pi * (diameter**4 - inner_diameter**4)) / 32
        
        torsional_stress = (torque * 1000 * (diameter/2)) / polar_moment  # Convert to N·mm
        
        # 2. Bending stress: σ = (M·y)/I
        if hollow_ratio == 0:
            moment_of_inertia = (np.pi * diameter**4) / 64
        else:
            moment_of_inertia = (np.pi * (diameter**4 - inner_diameter**4)) / 64
        
        bending_stress = (bending_moment * 1000 * (diameter/2)) / moment_of_inertia
        
        # 3. Combined stress (Von Mises): σ_eq = √(σ² + 3τ²)
        equivalent_stress = np.sqrt(bending_stress**2 + 3 * torsional_stress**2)
        
        # 4. Safety factor
        safety_factor = yield_strength / (equivalent_stress / 1e6) if equivalent_stress > 0 else 10
        safety_factor = min(safety_factor, 10)  # Cap at 10
        
        # 5. Critical speed (for rotating shafts)
        # ω_critical = √(g·E·I/(m·L³))
        mass_per_length = density * np.pi * (diameter**2 - inner_diameter**2) / 4 / 1e6  # kg/mm
        if length > 0 and mass_per_length > 0:
            critical_speed = np.sqrt((9.81 * elastic_modulus * 1e9 * moment_of_inertia * 1e-12) / 
                                    (mass_per_length * (length/1000)**3)) * 30/np.pi  # Convert to RPM
        else:
            critical_speed = 10000
        
        # 6. Deflection under load
        # δ = (F·L³)/(3·E·I)
        deflection = (axial_force * (length/1000)**3) / (3 * elastic_modulus * 1e9 * moment_of_inertia * 1e-12) * 1000  # mm
        
        # 7. Thermal expansion
        thermal_growth = diameter * thermal_expansion * (temperature - 20)
        
        # ============================================================
        # OPTIMIZED OUTPUT PARAMETERS
        # ============================================================
        
        # Optimize diameter based on stress requirements (safety factor target = 2-3)
        if safety_factor < 2:
            diameter_opt = diameter * 1.2  # Increase diameter if under-designed
        elif safety_factor > 4:
            diameter_opt = diameter * 0.95  # Reduce diameter if over-designed
        else:
            diameter_opt = diameter
        
        # Optimize length considering deflection and critical speed
        if deflection > 0.001 * length:  # Deflection > 0.1% of length
            length_opt = length * 0.9
        elif rpm > 0.7 * critical_speed:  # Too close to critical speed
            length_opt = length * 0.85
        else:
            length_opt = length
        
        # Optimize hollow ratio for weight reduction while maintaining strength
        if safety_factor > 3 and hollow_ratio == 0:
            hollow_ratio_opt = 0.5  # Suggest hollow shaft
        elif safety_factor < 2 and hollow_ratio > 0.5:
            hollow_ratio_opt = max(0, hollow_ratio - 0.2)  # Reduce hollow ratio
        else:
            hollow_ratio_opt = hollow_ratio
        
        # Optimize wall thickness for hollow shafts
        if hollow_ratio_opt > 0:
            wall_thickness_opt = (diameter_opt / 2) * (1 - hollow_ratio_opt)
        else:
            wall_thickness_opt = diameter_opt / 2
        
        # Recommended surface finish (Ra in micrometers) based on application
        surface_finish_opt = {
            "Transmission": 1.6,
            "PowerTransmission": 0.8,
            "Spindle": 0.4,
            "Axle": 3.2
        }[application]
        
        data.append([
            # Input features
            length, diameter, hollow_ratio, material, application,
            torque, bending_moment, axial_force, rpm, temperature,
            
            # Optimized outputs
            diameter_opt, length_opt, hollow_ratio_opt, 
            wall_thickness_opt, surface_finish_opt,
            safety_factor, equivalent_stress
        ])

    cols = [
        # Inputs
        "length", "diameter", "hollow_ratio", "material", "application",
        "torque", "bending_moment", "axial_force", "rpm", "temperature",
        
        # Outputs
        "diameter_opt", "length_opt", "hollow_ratio_opt",
        "wall_thickness_opt", "surface_finish_opt",
        "safety_factor", "equivalent_stress"
    ]

    return pd.DataFrame(data, columns=cols)


# ======================================================================
# 2️⃣ Generate Dataset
# ======================================================================

print("🔧 Generating shaft design dataset...")
df = generate_shaft_data(n_samples=60000)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}\n")


# ======================================================================
# 3️⃣ Split Features / Targets
# ======================================================================

X = df[[
    "length", "diameter", "hollow_ratio", "material", "application",
    "torque", "bending_moment", "axial_force", "rpm", "temperature"
]]

y = df[[
    "diameter_opt", "length_opt", "hollow_ratio_opt",
    "wall_thickness_opt", "surface_finish_opt"
]]


# ======================================================================
# 4️⃣ Preprocessing
# ======================================================================

cat_features = ["material", "application"]
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

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")


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

print(f"Processed feature dimensions: {X_train_proc.shape[1]}\n")


# ======================================================================
# 8️⃣ Neural Network Model
# ======================================================================

print("🧠 Building neural network model...")

model = Sequential([
    Dense(256, input_dim=X_train_proc.shape[1], activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.1),
    
    Dense(32, activation="relu"),
    
    Dense(y_train_scaled.shape[1], activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

print(model.summary())

es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

print("\n🚀 Starting training...\n")

history = model.fit(
    X_train_proc, y_train_scaled,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    callbacks=[es],
    verbose=1
)


# ======================================================================
# 9️⃣ Evaluate Model
# ======================================================================

print("\n📊 Evaluating model performance...\n")

pred_scaled = model.predict(X_test_proc)
pred = scaler_y.inverse_transform(pred_scaled)

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("=" * 60)
print("📈 FINAL MODEL EVALUATION")
print("=" * 60)
print(f"R² Score: {r2:.6f}")
print(f"RMSE: {rmse:.4f}")

# Per-parameter R² scores
for i, col in enumerate(y.columns):
    r2_param = r2_score(y_test.iloc[:, i], pred[:, i])
    print(f"R² for {col}: {r2_param:.4f}")
print("=" * 60 + "\n")


# ======================================================================
# 🔟 PATTERN-LEARNING VERIFICATION TESTS
# ======================================================================

print("🔍 Running Pattern-Learning Verification Tests...\n")

# A) True vs Predicted Scatter Plots
cols = y.columns.tolist()
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, col in enumerate(cols):
    axs[i].scatter(y_test.iloc[:, i], pred[:, i], s=10, alpha=0.5)
    axs[i].plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], "r--", linewidth=2)
    axs[i].set_xlabel("True Value")
    axs[i].set_ylabel("Predicted Value")
    axs[i].set_title(f"{col}\nR² = {r2_score(y_test.iloc[:, i], pred[:, i]):.4f}")
    axs[i].grid(True, alpha=0.3)

# Hide unused subplot
axs[5].axis('off')

plt.tight_layout()
plt.savefig("shaft_prediction_scatter.png", dpi=150)
print("✅ Saved: shaft_prediction_scatter.png")
plt.show()

# B) Error Distribution Plot
errors = y_test.values - pred
plt.figure(figsize=(10, 6))
plt.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution (Should be centered at 0)")
plt.grid(True, alpha=0.3)
plt.savefig("shaft_error_distribution.png", dpi=150)
print("✅ Saved: shaft_error_distribution.png")
plt.show()

# C) Physics Sensitivity Test - Diameter vs Torque
print("\n⚙️ Testing physics sensitivity: Diameter optimization with varying torque...")

sample = X_test.iloc[100:101].copy()
torque_values = []
diameter_predictions = []

for t in range(500, 10000, 500):
    sample["torque"] = t
    xproc = preprocessor.transform(sample)
    p = scaler_y.inverse_transform(model.predict(xproc, verbose=0))
    torque_values.append(t)
    diameter_predictions.append(p[0][0])

plt.figure(figsize=(10, 6))
plt.plot(torque_values, diameter_predictions, marker='o', linewidth=2)
plt.xlabel("Applied Torque (N·m)")
plt.ylabel("Optimized Diameter (mm)")
plt.title("Physics Consistency: Diameter increases with Torque")
plt.grid(True, alpha=0.3)
plt.savefig("shaft_physics_sensitivity.png", dpi=150)
print("✅ Saved: shaft_physics_sensitivity.png")
plt.show()


# ======================================================================
# 1️⃣1️⃣ Save Model + Preprocessor + Scaler
# ======================================================================

BASE_DIR = r"C:\Users\Sanika\Desktop\IEEE 2\AutoCADify"
os.makedirs(BASE_DIR, exist_ok=True)

model_path = os.path.join(BASE_DIR, "shaft_enhancer_model.h5")
preprocessor_path = os.path.join(BASE_DIR, "shaft_preprocessor.pkl")
scaler_path = os.path.join(BASE_DIR, "shaft_scaler_y.pkl")

model.save(model_path)
joblib.dump(preprocessor, preprocessor_path)
joblib.dump(scaler_y, scaler_path)

print("\n" + "=" * 60)
print("✅ MODEL AND SCALERS SAVED SUCCESSFULLY")
print("=" * 60)
print(f"📁 Model: {model_path}")
print(f"📁 Preprocessor: {preprocessor_path}")
print(f"📁 Scaler: {scaler_path}")
print("=" * 60)

print("\n🎉 Training complete! You can now run the Streamlit app.")
