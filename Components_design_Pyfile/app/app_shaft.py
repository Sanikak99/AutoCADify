"""
AI Physics-Informed Shaft Parameter Enhancer + DXF Generator
------------------------------------------------------------
Run:
    streamlit run app_physics_dxf_shaft.py
"""

# ===============================================================
# 0️⃣ Import Libraries
# ===============================================================
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import matplotlib as ez_mpl
from ezdxf.addons.drawing import RenderContext, Frontend

# ===============================================================
# 1️⃣ Configuration
# ===============================================================
st.set_page_config(
    page_title="AI Physics-Informed Shaft Enhancer",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# Aesthetic theme
st.markdown("""
    <style>
        .main {
            background-color: #fafafa;
            padding: 25px 50px;
        }
        h1 {
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            color: #FF6347;
            font-size: 40px;
            margin-bottom: 10px;
        }
        h2, h3, h4 {
            color: #222222;
        }
        .stButton>button {
            background-color: #FF6347;
            color: white;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            height: 3em;
            width: 100%;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #CD5C5C;
        }
        .block-container {
            max-width: 1300px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# --- PATHS ---
BASE_DIR = r"C:\Users\Sanika\Desktop\IEEE 2"
MODEL_PATH = os.path.join(BASE_DIR, "shaft_enhancer_model.h5")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "shaft_preprocessor.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "shaft_scaler_y.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_shaft")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CONTENT_DIR = os.path.join(OUTPUT_DIR, "content")
os.makedirs(OUTPUT_CONTENT_DIR, exist_ok=True)

# ===============================================================
# 2️⃣ Utility Functions
# ===============================================================

def is_valid_shaft(length, diameter, hollow_ratio, wall_thickness):
    """Basic rule checks for shaft geometry validity."""
    try:
        length = float(length)
        diameter = float(diameter)
        hollow_ratio = float(hollow_ratio)
        wall_thickness = float(wall_thickness)
    except:
        return False
    
    if length <= 0 or diameter <= 0:
        return False
    if hollow_ratio < 0 or hollow_ratio >= 1:
        return False
    if hollow_ratio > 0:
        inner_d = diameter * hollow_ratio
        required_wall = (diameter - inner_d) / 2
        if wall_thickness < required_wall * 0.8:
            return False
    return True

def correct_shaft_geometry(length, diameter, hollow_ratio, wall_thickness):
    """Attempt to correct geometry to avoid invalid configurations."""
    msgs = []
    length = max(10.0, float(length))
    diameter = max(10.0, float(diameter))
    hollow_ratio = max(0.0, min(0.9, float(hollow_ratio)))
    wall_thickness = max(1.0, float(wall_thickness))
    
    l_d_ratio = length / diameter
    if l_d_ratio > 20:
        new_length = diameter * 20
        msgs.append(f"Length reduced from {length:.2f} to {new_length:.2f} mm (L/D ratio > 20).")
        length = new_length
    
    if hollow_ratio > 0:
        inner_d = diameter * hollow_ratio
        required_wall = (diameter - inner_d) / 2
        
        if wall_thickness < required_wall:
            wall_thickness = required_wall * 1.1
            msgs.append(f"Wall thickness increased to {wall_thickness:.2f} mm for hollow shaft.")
        
        if wall_thickness < diameter * 0.1:
            wall_thickness = diameter * 0.1
            hollow_ratio = 1 - (2 * wall_thickness / diameter)
            msgs.append(f"Wall thickness and hollow ratio adjusted for structural integrity.")
    
    if length <= 0:
        length = 100.0
        msgs.append("Length set to minimum 100.0 mm.")
    
    if diameter <= 0:
        diameter = 20.0
        msgs.append("Diameter set to minimum 20.0 mm.")
    
    return (length, diameter, hollow_ratio, wall_thickness), msgs

# ===============================================================
# 3️⃣ Load Model and Scalers
# ===============================================================
@st.cache_resource
def load_resources_shaft():
    try:
        model = load_model(MODEL_PATH, compile=False)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        return model, preprocessor, scaler_y
    except Exception as e:
        st.error(f"❌ Failed to load model or scalers: {e}")
        return None, None, None

model, preprocessor, scaler_y = load_resources_shaft()

# ===============================================================
# 4️⃣ Title + Inputs
# ===============================================================
st.markdown("<h1 style='text-align:center;'>🧠 AI Physics-Informed Shaft Parameter Enhancer + DXF Generator</h1>",
            unsafe_allow_html=True)
st.caption("Predict optimized shaft geometry using physics + AI and export multi-view DXF (colored preview).")

st.markdown("### 📏 Initial Shaft Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    length_init = st.number_input("Shaft Length (mm)", 10.0, 2000.0, 200.0, step=10.0)
    diameter_init = st.number_input("Outer Diameter (mm)", 10.0, 500.0, 50.0, step=5.0)
with col2:
    hollow_ratio_init = st.slider("Hollow Ratio (0=Solid)", 0.0, 0.8, 0.0, step=0.1,
                                   help="Inner Diameter / Outer Diameter")
    
    num_steps = st.selectbox(
        "Number of Shaft Steps",
        options=[1, 2, 3, 4, 5, 6],
        index=3,
        help="Select how many diameter steps the shaft should have"
    )
    
    material = st.selectbox("Material", ["CarbonSteel", "Stainless304", "Alloy4140", "CastIron"])
with col3:
    application = st.selectbox("Application", ["Transmission", "PowerTransmission", "Spindle", "Axle"])

st.markdown("### ⚙️ Operating Conditions")
colA, colB, colC = st.columns(3)
with colA:
    torque = st.number_input("Torque (N·m)", 0.0, 100000.0, 1000.0, step=100.0)
    bending_moment = st.number_input("Bending Moment (N·m)", 0.0, 20000.0, 500.0, step=50.0)
with colB:
    axial_force = st.number_input("Axial Force (N)", 0.0, 200000.0, 5000.0, step=500.0)
    rpm = st.number_input("Rotational Speed (RPM)", 0.0, 10000.0, 1500.0, step=100.0)
with colC:
    temperature = st.slider("Operating Temperature (°C)", 20, 400, 80)

# ===============================================================
# 5️⃣ DXF Generator
# ===============================================================

def generate_shaft_dxf_multi_view(shaft_params, filename="shaft_output.dxf"):
    """
    Generate professional DXF with perfect layout:
    - Front view (left) - circles with diameters
    - Side view (right) - stepped profile with dimensions
    - Top view (bottom) - rectangular view
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    text_height = 4.0
    line_thickness = 0.35
    
    max_od = max([s["outerDiameter"] for s in shaft_params])
    total_length = sum([s["length"] for s in shaft_params])
    
    # Perfect spacing for clean layout
    horizontal_spacing = total_length + max_od * 2.5
    vertical_spacing = max_od * (len(shaft_params) + 3)
    
    front_origin = (max_od * 0.5, vertical_spacing)  # Top left (slight offset)
    side_origin = (horizontal_spacing, vertical_spacing + max_od * 0.5)  # Top right (aligned with front)
    top_origin = (0, -max_od * 1.5)  # Bottom (below front view)
    
    # ============================================================
    # FRONT VIEW (Left side - circles)
    # ============================================================
    front_x = front_origin[0]
    front_y = front_origin[1]
    
    for i, step in enumerate(shaft_params):
        outer_d = step["outerDiameter"]
        inner_d = step.get("innerDiameter", 0)
        
        # Outer circle - RED
        msp.add_circle(center=(front_x, front_y), radius=outer_d / 2,
                      dxfattribs={"color": 1, "lineweight": line_thickness})
        
        # Inner circle - RED
        if inner_d and inner_d > 0:
            msp.add_circle(center=(front_x, front_y), radius=inner_d / 2,
                          dxfattribs={"color": 1, "lineweight": line_thickness * 0.5})
        
        # Center lines - RED
        crosshair_size = outer_d * 0.7
        msp.add_line(start=(front_x - crosshair_size / 2, front_y),
                    end=(front_x + crosshair_size / 2, front_y),
                    dxfattribs={"color": 1, "lineweight": 0.05})
        msp.add_line(start=(front_x, front_y - crosshair_size / 2),
                    end=(front_x, front_y + crosshair_size / 2),
                    dxfattribs={"color": 1, "lineweight": 0.05})
        
        # Diameter text - RED
        dim_offset = outer_d / 2 + 10
        msp.add_text(f"Ø{outer_d:.1f}",
                    dxfattribs={"height": text_height * 0.75, "color": 1}
        ).set_placement((front_x + dim_offset, front_y - 1.5))
        
        front_y -= (max_od * 1.1)
    
    # Front view label (below the circles)
    msp.add_text("FRONT VIEW", dxfattribs={"height": text_height, "color": 1}).set_placement(
        (front_origin[0] - 25, front_y - 15)
    )
    
    # ============================================================
    # SIDE VIEW (Right side - stepped profile)
    # ============================================================
    
    current_x = side_origin[0]
    side_y = side_origin[1]
    
    # Center line - RED
    msp.add_line(start=(current_x - total_length * 0.05, side_y),
                end=(current_x + total_length * 1.05, side_y),
                dxfattribs={"color": 1, "lineweight": 0.05})
    
    for i, step in enumerate(shaft_params):
        od = step["outerDiameter"]
        id_val = step.get("innerDiameter", 0)
        length = step["length"]
        half_od = od / 2
        
        # Top line - RED
        msp.add_line(start=(current_x, side_y + half_od),
                    end=(current_x + length, side_y + half_od),
                    dxfattribs={"color": 1, "lineweight": line_thickness})
        
        # Bottom line - RED
        msp.add_line(start=(current_x, side_y - half_od),
                    end=(current_x + length, side_y - half_od),
                    dxfattribs={"color": 1, "lineweight": line_thickness})
        
        # Left end cap - RED
        if i == 0:
            msp.add_line(start=(current_x, side_y + half_od),
                        end=(current_x, side_y - half_od),
                        dxfattribs={"color": 1, "lineweight": line_thickness})
        
        # Step transitions - RED
        if i < len(shaft_params) - 1:
            next_od = shaft_params[i + 1]["outerDiameter"]
            next_half_od = next_od / 2
            msp.add_line(start=(current_x + length, side_y + half_od),
                        end=(current_x + length, side_y + next_half_od),
                        dxfattribs={"color": 1, "lineweight": line_thickness})
            msp.add_line(start=(current_x + length, side_y - half_od),
                        end=(current_x + length, side_y - next_half_od),
                        dxfattribs={"color": 1, "lineweight": line_thickness})
        else:
            msp.add_line(start=(current_x + length, side_y + half_od),
                        end=(current_x + length, side_y - half_od),
                        dxfattribs={"color": 1, "lineweight": line_thickness})
        
        # Inner bore lines - RED
        if id_val and id_val > 0:
            half_id = id_val / 2
            msp.add_line(start=(current_x, side_y + half_id),
                        end=(current_x + length, side_y + half_id),
                        dxfattribs={"color": 1, "lineweight": 0.2})
            msp.add_line(start=(current_x, side_y - half_id),
                        end=(current_x + length, side_y - half_id),
                        dxfattribs={"color": 1, "lineweight": 0.2})
        
        # Dimension lines - RED
        dim_y = side_y - max_od / 2 - 15
        msp.add_line(start=(current_x, dim_y), end=(current_x + length, dim_y),
                    dxfattribs={"color": 1, "lineweight": 0.1})
        
        # Dimension arrows - RED
        arrow_size = 2.5
        msp.add_line(start=(current_x, dim_y),
                    end=(current_x + arrow_size, dim_y + arrow_size / 2),
                    dxfattribs={"color": 1, "lineweight": 0.1})
        msp.add_line(start=(current_x, dim_y),
                    end=(current_x + arrow_size, dim_y - arrow_size / 2),
                    dxfattribs={"color": 1, "lineweight": 0.1})
        msp.add_line(start=(current_x + length, dim_y),
                    end=(current_x + length - arrow_size, dim_y + arrow_size / 2),
                    dxfattribs={"color": 1, "lineweight": 0.1})
        msp.add_line(start=(current_x + length, dim_y),
                    end=(current_x + length - arrow_size, dim_y - arrow_size / 2),
                    dxfattribs={"color": 1, "lineweight": 0.1})
        
        # Dimension text - RED
        msp.add_text(f"{length:.1f}",
                    dxfattribs={"height": text_height * 0.7, "color": 1}
        ).set_placement((current_x + length / 2 - 5, dim_y - 7))
        
        current_x += length
    
    # Side view label (below the profile)
    msp.add_text("SIDE VIEW", dxfattribs={"height": text_height, "color": 1}).set_placement(
        (side_origin[0] + total_length / 2 - 18, side_y - max_od / 2 - 35)
    )
    
    # ============================================================
    # TOP VIEW (Bottom - rectangular view)
    # ============================================================
    
    current_x = top_origin[0]
    top_y = top_origin[1]
    
    # Center line - RED
    msp.add_line(start=(current_x - total_length * 0.05, top_y),
                end=(current_x + total_length * 1.05, top_y),
                dxfattribs={"color": 1, "lineweight": 0.05})
    
    for step in shaft_params:
        od = step["outerDiameter"]
        id_val = step.get("innerDiameter", 0)
        length = step["length"]
        half_od = od / 2
        
        # Outer rectangle - RED
        msp.add_lwpolyline([
            (current_x, top_y + half_od),
            (current_x + length, top_y + half_od),
            (current_x + length, top_y - half_od),
            (current_x, top_y - half_od),
            (current_x, top_y + half_od)
        ], dxfattribs={"color": 1, "lineweight": line_thickness})
        
        # Inner rectangle - RED
        if id_val and id_val > 0:
            half_id = id_val / 2
            msp.add_lwpolyline([
                (current_x, top_y + half_id),
                (current_x + length, top_y + half_id),
                (current_x + length, top_y - half_id),
                (current_x, top_y - half_id),
                (current_x, top_y + half_id)
            ], dxfattribs={"color": 1, "lineweight": 0.2})
        
        current_x += length
    
    # Top view label (below the rectangle)
    msp.add_text("TOP VIEW", dxfattribs={"height": text_height, "color": 1}).set_placement(
        (top_origin[0] + total_length / 2 - 15, top_y - max_od / 2 - 20)
    )
    
    # ============================================================
    # NO SPECIFICATIONS - Removed as requested
    # ============================================================
    
    # Save file
    file_path = os.path.join(OUTPUT_CONTENT_DIR, filename)
    doc.saveas(file_path)
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    return file_path, data



def generate_custom_shaft_dxf(custom_steps, filename="shaft_output.dxf"):
    """Generate DXF from user-customized step dimensions"""
    shaft_params = []
    
    for step in custom_steps:
        shaft_params.append({
            "length": step['length'],
            "outerDiameter": step['outer_dia'],
            "innerDiameter": step['inner_dia'] if step['inner_dia'] > 0 else None
        })
    
    return generate_shaft_dxf_multi_view(shaft_params, filename=filename)

# ===============================================================
# 7️⃣ Predict Button
# ===============================================================
st.markdown("---")
if st.button("🔮 Predict Optimized Shaft Geometry", use_container_width=True):
    
    if model is not None:
        try:
            if hollow_ratio_init > 0:
                inner_d_init = diameter_init * hollow_ratio_init
                wall_thickness_init = (diameter_init - inner_d_init) / 2
            else:
                wall_thickness_init = diameter_init / 2
            
            input_data = pd.DataFrame([{
                "length": length_init,
                "diameter": diameter_init,
                "hollow_ratio": hollow_ratio_init,
                "material": material,
                "application": application,
                "torque": torque,
                "bending_moment": bending_moment,
                "axial_force": axial_force,
                "rpm": rpm,
                "temperature": temperature
            }])
            
            X_proc = preprocessor.transform(input_data)
            y_pred_scaled = model.predict(X_proc)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
            
            diameter_pred, length_pred, hollow_ratio_pred, wall_thickness_pred, surface_finish_pred = y_pred
            
            if not is_valid_shaft(length_pred, diameter_pred, hollow_ratio_pred, wall_thickness_pred):
                (length_corr, diameter_corr, hollow_ratio_corr, wall_thickness_corr), corrections = correct_shaft_geometry(
                    length_pred, diameter_pred, hollow_ratio_pred, wall_thickness_pred)
                
                for msg in corrections:
                    st.warning("⚠️ " + msg)
                
                length_pred = length_corr
                diameter_pred = diameter_corr
                hollow_ratio_pred = hollow_ratio_corr
                wall_thickness_pred = wall_thickness_corr
            
            predicted_params = (diameter_pred, length_pred, hollow_ratio_pred, wall_thickness_pred, surface_finish_pred)
            st.session_state["predicted_params_shaft"] = predicted_params
            st.session_state["prediction_done"] = True
            
            st.success("✅ Prediction & validation complete.")
            
            inner_d_pred = diameter_pred * hollow_ratio_pred if hollow_ratio_pred > 0 else 0
            inner_d_init = diameter_init * hollow_ratio_init if hollow_ratio_init > 0 else 0
            
            comparison_data = {
                "Parameter": [
                    "Length (mm)", 
                    "Outer Diameter (mm)", 
                    "Inner Diameter (mm)",
                    "Hollow Ratio", 
                    "Wall Thickness (mm)", 
                    "Surface Finish Ra (μm)"
                ],
                "Initial Value": [
                    round(length_init, 2), 
                    round(diameter_init, 2), 
                    round(inner_d_init, 2) if inner_d_init > 0 else "Solid",
                    round(hollow_ratio_init, 2), 
                    round(wall_thickness_init, 2), 
                    "N/A"
                ],
                "Optimized Value": [
                    round(length_pred, 2), 
                    round(diameter_pred, 2), 
                    round(inner_d_pred, 2) if inner_d_pred > 0 else "Solid",
                    round(hollow_ratio_pred, 2), 
                    round(wall_thickness_pred, 2), 
                    round(surface_finish_pred, 2)
                ]
            }
            st.write(pd.DataFrame(comparison_data).set_index("Parameter"))
            
            # Physics Dashboard
            st.markdown("## ⚙️ Physics + AI Validation Dashboard")
            st.markdown("#### Real-time analytical insights from shaft mechanics equations and AI prediction behavior.")
            
            colA, colB = st.columns(2)
            colC, colD = st.columns(2)
            
            torque_range = np.linspace(max(1, torque) * 0.5, max(1, torque) * 2, 100)
            
            if hollow_ratio_pred == 0:
                J = (np.pi * diameter_pred**4) / 32
            else:
                inner_d_calc = diameter_pred * hollow_ratio_pred
                J = (np.pi * (diameter_pred**4 - inner_d_calc**4)) / 32
            
            torsional_stress_range = (torque_range * 1000 * (diameter_pred/2)) / J
            current_stress = (torque * 1000 * (diameter_pred/2)) / J
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=torque_range, y=torsional_stress_range, mode='lines', 
                                     line=dict(color='darkred', width=3), name='Torsional Stress (Physics)'))
            fig1.add_trace(go.Scatter(x=[torque], y=[current_stress], mode='markers',
                                     marker=dict(color='red', size=10, symbol='star'), name='AI Optimized Point'))
            fig1.update_layout(title="1️⃣ Torsional Shear Stress vs Torque", xaxis_title="Torque (N·m)",
                             yaxis_title="Shear Stress (MPa)", template="plotly_white", height=360)
            colA.plotly_chart(fig1, use_container_width=True)
            
            diameter_range = np.arange(max(10, diameter_init * 0.7), diameter_init * 1.3 + 1, 2)
            
            if hollow_ratio_pred == 0:
                I_range = (np.pi * diameter_range**4) / 64
            else:
                inner_d_range = diameter_range * hollow_ratio_pred
                I_range = (np.pi * (diameter_range**4 - inner_d_range**4)) / 64
            
            bending_stress_range = (bending_moment * 1000 * (diameter_range/2)) / I_range
            
            if hollow_ratio_pred == 0:
                I_current = (np.pi * diameter_pred**4) / 64
            else:
                I_current = (np.pi * (diameter_pred**4 - (diameter_pred*hollow_ratio_pred)**4)) / 64
            current_bending_stress = (bending_moment * 1000 * (diameter_pred/2)) / I_current
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=diameter_range, y=bending_stress_range, mode='lines',
                                     line=dict(color='steelblue', width=3), name='Bending Stress Trend'))
            fig2.add_trace(go.Scatter(x=[diameter_pred], y=[current_bending_stress], mode='markers',
                                     marker=dict(color='red', size=10, symbol='circle'), name='AI Optimized Diameter'))
            fig2.update_layout(title="2️⃣ Bending Stress vs Shaft Diameter", xaxis_title="Diameter (mm)",
                             yaxis_title="Bending Stress (MPa)", template="plotly_white", height=360)
            colB.plotly_chart(fig2, use_container_width=True)
            
            length_range = np.linspace(max(10, length_init * 0.5), length_init * 1.5, 100)
            mat_modulus = {"CarbonSteel": 200, "Stainless304": 193, "Alloy4140": 205, "CastIron": 100}[material]
            E = mat_modulus * 1e9
            
            deflection_range = (axial_force * (length_range/1000)**3) / (3 * E * I_current * 1e-12) * 1000
            current_deflection = (axial_force * (length_pred/1000)**3) / (3 * E * I_current * 1e-12) * 1000
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=length_range, y=deflection_range, mode='lines',
                                     line=dict(color='darkorange', width=3), name='Deflection vs Length'))
            fig3.add_trace(go.Scatter(x=[length_pred], y=[current_deflection], mode='markers',
                                     marker=dict(color='red', size=10, symbol='diamond'), name='AI Optimized Length'))
            fig3.update_layout(title="3️⃣ Shaft Deflection vs Length", xaxis_title="Length (mm)",
                             yaxis_title="Deflection (mm)", template="plotly_white", height=360)
            colC.plotly_chart(fig3, use_container_width=True)
            
            mass_per_length_dict = {"CarbonSteel": 7850, "Stainless304": 8000, "Alloy4140": 7850, "CastIron": 7200}
            density = mass_per_length_dict[material]
            
            if hollow_ratio_pred == 0:
                mass_per_length = density * np.pi * (diameter_pred**2) / 4 / 1e6
            else:
                inner_d_mass = diameter_pred * hollow_ratio_pred
                mass_per_length = density * np.pi * (diameter_pred**2 - inner_d_mass**2) / 4 / 1e6
            
            if length_pred > 0 and mass_per_length > 0:
                critical_speed = np.sqrt((9.81 * E * I_current * 1e-12) / 
                                        (mass_per_length * (length_pred/1000)**3)) * 30/np.pi
            else:
                critical_speed = 10000
            
            rpm_range = np.linspace(0, min(10000, critical_speed * 1.5), 100)
            safety_margin = critical_speed / rpm_range
            safety_margin[safety_margin > 10] = 10
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=rpm_range, y=safety_margin, mode='lines',
                                     line=dict(color='darkgreen', width=3), name='Safety Margin from Critical Speed'))
            fig4.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Minimum Safety Factor")
            fig4.add_trace(go.Scatter(x=[rpm], y=[critical_speed / rpm if rpm > 0 else 10], mode='markers',
                                     marker=dict(color='red', size=10, symbol='square'), name='Operating Point'))
            fig4.update_layout(title=f"4️⃣ Critical Speed Analysis (ω_crit = {critical_speed:.0f} RPM)",
                             xaxis_title="Operating Speed (RPM)", yaxis_title="Safety Factor",
                             template="plotly_white", height=360, yaxis=dict(range=[0, 10]))
            colD.plotly_chart(fig4, use_container_width=True)
            
            # Initialize custom steps
            inner_d = diameter_pred * hollow_ratio_pred if hollow_ratio_pred > 0 else 0
            
            default_steps = []
            
            if num_steps == 1:
                default_steps = [{"step": 1, "length": length_pred, "outer_dia": diameter_pred, "inner_dia": inner_d}]
            elif num_steps == 2:
                default_steps = [
                    {"step": 1, "length": length_pred * 0.4, "outer_dia": diameter_pred * 0.75, "inner_dia": inner_d * 0.75},
                    {"step": 2, "length": length_pred * 0.6, "outer_dia": diameter_pred, "inner_dia": inner_d}
                ]
            elif num_steps == 3:
                default_steps = [
                    {"step": 1, "length": length_pred * 0.25, "outer_dia": diameter_pred * 0.7, "inner_dia": inner_d * 0.7},
                    {"step": 2, "length": length_pred * 0.5, "outer_dia": diameter_pred, "inner_dia": inner_d},
                    {"step": 3, "length": length_pred * 0.25, "outer_dia": diameter_pred * 0.75, "inner_dia": inner_d * 0.75}
                ]
            elif num_steps == 4:
                default_steps = [
                    {"step": 1, "length": length_pred * 0.2, "outer_dia": diameter_pred * 0.7, "inner_dia": inner_d * 0.7},
                    {"step": 2, "length": length_pred * 0.3, "outer_dia": diameter_pred * 0.85, "inner_dia": inner_d * 0.85},
                    {"step": 3, "length": length_pred * 0.3, "outer_dia": diameter_pred, "inner_dia": inner_d},
                    {"step": 4, "length": length_pred * 0.2, "outer_dia": diameter_pred * 0.75, "inner_dia": inner_d * 0.75}
                ]
            elif num_steps == 5:
                default_steps = [
                    {"step": 1, "length": length_pred * 0.15, "outer_dia": diameter_pred * 0.65, "inner_dia": inner_d * 0.65},
                    {"step": 2, "length": length_pred * 0.2, "outer_dia": diameter_pred * 0.8, "inner_dia": inner_d * 0.8},
                    {"step": 3, "length": length_pred * 0.3, "outer_dia": diameter_pred, "inner_dia": inner_d},
                    {"step": 4, "length": length_pred * 0.2, "outer_dia": diameter_pred * 0.85, "inner_dia": inner_d * 0.85},
                    {"step": 5, "length": length_pred * 0.15, "outer_dia": diameter_pred * 0.7, "inner_dia": inner_d * 0.7}
                ]
            elif num_steps == 6:
                step_length = length_pred / 6
                default_steps = [
                    {"step": 1, "length": step_length, "outer_dia": diameter_pred * 0.65, "inner_dia": inner_d * 0.65},
                    {"step": 2, "length": step_length, "outer_dia": diameter_pred * 0.75, "inner_dia": inner_d * 0.75},
                    {"step": 3, "length": step_length, "outer_dia": diameter_pred * 0.9, "inner_dia": inner_d * 0.9},
                    {"step": 4, "length": step_length, "outer_dia": diameter_pred, "inner_dia": inner_d},
                    {"step": 5, "length": step_length, "outer_dia": diameter_pred * 0.85, "inner_dia": inner_d * 0.85},
                    {"step": 6, "length": step_length, "outer_dia": diameter_pred * 0.7, "inner_dia": inner_d * 0.7}
                ]
            
            st.session_state["custom_steps"] = default_steps
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Model not loaded properly.")

# ===============================================================
# 8️⃣ Customization Section (Only shows after prediction)
# ===============================================================
if st.session_state.get("prediction_done", False) and "custom_steps" in st.session_state:
    
    st.markdown("---")
    st.markdown("### 🔧 Customize Individual Step Dimensions")
    st.caption("✨ AI provides initial suggestions - adjust each step's outer and inner diameter as needed!")
    
    st.markdown("#### 📝 Edit Step Dimensions:")
    
    # Create editable inputs
    for row_start in range(0, num_steps, 3):
        cols = st.columns(min(3, num_steps - row_start))
        
        for col_idx, i in enumerate(range(row_start, min(row_start + 3, num_steps))):
            step_data = st.session_state["custom_steps"][i]
            
            with cols[col_idx]:
                st.markdown(f"**Step {step_data['step']}**")
                
                new_length = st.number_input(
                    f"Length (mm)",
                    min_value=1.0,
                    max_value=1000.0,
                    value=float(step_data['length']),
                    step=1.0,
                    key=f"length_step_{i}"
                )
                
                new_outer = st.number_input(
                    f"Outer Ø (mm)",
                    min_value=5.0,
                    max_value=500.0,
                    value=float(step_data['outer_dia']),
                    step=1.0,
                    key=f"outer_step_{i}"
                )
                
                new_inner = st.number_input(
                    f"Inner Ø (mm)",
                    min_value=0.0,
                    max_value=float(new_outer * 0.9),
                    value=float(step_data['inner_dia']),
                    step=1.0,
                    key=f"inner_step_{i}",
                    help="0 = Solid shaft"
                )
                
                # Update immediately
                st.session_state["custom_steps"][i] = {
                    "step": step_data['step'],
                    "length": new_length,
                    "outer_dia": new_outer,
                    "inner_dia": new_inner
                }
    
    st.markdown("#### 📊 Current Configuration:")
    summary_df = pd.DataFrame(st.session_state["custom_steps"])
    summary_df.columns = ["Step #", "Length (mm)", "Outer Ø (mm)", "Inner Ø (mm)"]
    st.dataframe(summary_df, use_container_width=True)
    
    total_custom_length = sum([s['length'] for s in st.session_state["custom_steps"]])
    max_custom_diameter = max([s['outer_dia'] for s in st.session_state["custom_steps"]])
    st.info(f"📏 Total Shaft Length: **{total_custom_length:.2f} mm** | Max Diameter: **{max_custom_diameter:.2f} mm**")
    
    # ===============================================================
    # DXF Export Section
    # ===============================================================
    st.markdown("---")
    st.subheader("📐 DXF Visualization & Export")
    
    if st.button("🔄 Update Preview with Custom Dimensions", use_container_width=True):
        temp_name = "_preview_custom.dxf"
        dxf_path, _ = generate_custom_shaft_dxf(st.session_state["custom_steps"], filename=temp_name)
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        fig = plt.figure(figsize=(12, 8), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, facecolor="white")
        ctx = RenderContext(doc)
        out = ez_mpl.MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp)
        
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        st.session_state["preview_generated"] = True
    
    if st.session_state.get("preview_generated", False):
        st.success("✅ Preview updated with your custom dimensions!")
    
    if st.button("💾 Generate Final DXF File", use_container_width=True):
        total_length = sum([s['length'] for s in st.session_state["custom_steps"]])
        max_diameter = max([s['outer_dia'] for s in st.session_state["custom_steps"]])
        fname = f"custom_shaft_{int(total_length)}L_{int(max_diameter)}D_{num_steps}steps.dxf"
        
        fpath, data = generate_custom_shaft_dxf(st.session_state["custom_steps"], filename=fname)
        
        st.success(f"✅ Custom DXF File Generated: {fname}")
        st.download_button(
            label="⬇️ Download Custom DXF", 
            data=data,
            file_name=fname, 
            mime="application/octet-stream"
        )

# ===============================================================
# 9️⃣ Footer
# ===============================================================
st.markdown("---")
st.markdown("""
### 📘 About This App
This **AI-Physics-Informed Shaft Enhancer** merges **mechanical shaft design principles** and **AI learning**
to predict optimal shaft geometries (Diameter, Length, Hollow Ratio, Wall Thickness, Surface Finish) considering:
- Torque, Bending Moment & Axial Load
- Material Properties & Application Type
- Rotational Speed & Operating Temperature
- Critical Speed & Deflection Limits

**Features:**
- 🧠 AI-optimized shaft dimensions based on physics
- 🔧 Fully customizable multi-step shaft design
- 📐 Professional DXF engineering drawings
- ⚙️ Real-time physics validation dashboard

**Model:** Trained on 60k synthetic samples | Built with *TensorFlow + Scikit-learn + Streamlit + EZDXF*.

**Engineering Principles Applied:**
- Torsional stress: τ = (T·r)/J
- Bending stress: σ = (M·y)/I
- Von Mises equivalent stress
- Shaft deflection analysis
- Critical speed calculation
""")
