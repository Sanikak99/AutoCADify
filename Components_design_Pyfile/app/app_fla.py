"""
AI Physics-Informed Flange Parameter Enhancer + DXF Generator
------------------------------------------------------------
Run:
    streamlit run app_physics_dxf_flange.py
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
    page_title="AI Physics-Informed Flange Enhancer",
    layout="wide",  # ✅ full-screen professional view
    page_icon="🔩",
    initial_sidebar_state="expanded"
)

# Aesthetic theme (IEEE-style clean look)
st.markdown("""
    <style>
        .main {
            background-color: #fafafa;
            padding: 25px 50px;
        }
        h1 {
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            color: #1E90FF; /* Blue for Flange */
            font-size: 40px;
            margin-bottom: 10px;
        }
        h2, h3, h4 {
            color: #222222;
        }
        .stButton>button {
            background-color: #1E90FF; /* Blue */
            color: white;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            height: 3em;
            width: 100%;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #104E8B;
        }
        .block-container {
            max-width: 1300px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)


# --- IMPORTANT: PATHS MUST MATCH THE TRAINING SCRIPT'S SAVE LOCATION ---
BASE_DIR = r"C:\Users\Sanika\Documents\Project III\IEEE"
MODEL_PATH = os.path.join(BASE_DIR, "flange_enhancer_model.h5")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "flange_preprocessor.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "flange_scaler_y.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_flange")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure content subfolder exists for saved DXFs
OUTPUT_CONTENT_DIR = os.path.join(OUTPUT_DIR, "content")
os.makedirs(OUTPUT_CONTENT_DIR, exist_ok=True)


# ===============================================================
# 2️⃣ Utility Functions (validation, correction)
# ===============================================================

def is_valid_flange(inner_d, outer_d, thickness, bolt_circle_d, bolt_d, num_bolts):
    """Basic rule checks for flange geometry validity (pre-correction)."""
    try:
        inner_d = float(inner_d); outer_d = float(outer_d)
        thickness = float(thickness); bolt_circle_d = float(bolt_circle_d)
        bolt_d = float(bolt_d); num_bolts = int(round(num_bolts))
    except:
        return False

    if not (outer_d > inner_d): return False
    if not (inner_d < bolt_circle_d < outer_d): return False
    if num_bolts < 4 or num_bolts % 4 != 0: return False
    
    min_spacing = np.pi * bolt_circle_d / num_bolts
    if not (bolt_d * 1.5 < min_spacing): return False # Safety margin on spacing

    return True

def correct_flange_geometry(inner_d, outer_d, thickness, bolt_circle_d, bolt_d, num_bolts):
    """
    Attempt to correct geometry to avoid overlaps/protrusion:
    Returns corrected tuple and list of messages describing corrections.
    """
    msgs = []
    inner_d = float(inner_d); outer_d = float(outer_d)
    thickness = float(thickness); bolt_circle_d = float(bolt_circle_d)
    bolt_d = float(bolt_d); num_bolts = max(4, int(round(num_bolts)))

    # 1. Outer > Inner check
    if outer_d <= inner_d + 10:
        new_outer = inner_d + 50.0 # Min increase
        msgs.append(f"Outer diameter increased {outer_d:.2f} → {new_outer:.2f} (must be > inner).")
        outer_d = new_outer

    # 2. Bolt Circle Check (must be between inner and outer)
    if bolt_circle_d <= inner_d + 5:
        new_bolt_circle = inner_d + 15
        msgs.append(f"Bolt circle D increased {bolt_circle_d:.2f} → {new_bolt_circle:.2f} (must be > inner).")
        bolt_circle_d = new_bolt_circle
    if bolt_circle_d >= outer_d - 5:
        new_bolt_circle = outer_d - 15
        msgs.append(f"Bolt circle D reduced {bolt_circle_d:.2f} → {new_bolt_circle:.2f} (must be < outer).")
        bolt_circle_d = new_bolt_circle
    
    # Re-evaluate outer based on adjusted bolt circle
    if outer_d < bolt_circle_d + 15:
        new_outer = bolt_circle_d + 20
        msgs.append(f"Outer diameter increased {outer_d:.2f} → {new_outer:.2f} to fit the bolt circle.")
        outer_d = new_outer

    # 3. Number of Bolts Check (must be an even, typical multiple of 4, and fit)
    if num_bolts % 4 != 0 or num_bolts < 4:
        new_num_bolts = max(4, int(np.ceil(num_bolts / 4.0) * 4))
        msgs.append(f"Number of bolts adjusted {num_bolts} → {new_num_bolts} (multiple of 4).")
        num_bolts = new_num_bolts

    # 4. Bolt Spacing Check
    min_spacing_req = bolt_d * 1.5
    actual_spacing = np.pi * bolt_circle_d / num_bolts
    max_num_bolts = int(np.floor(np.pi * bolt_circle_d / min_spacing_req))
    
    if num_bolts > max_num_bolts:
        # Correct by reducing number of bolts (must be mult of 4)
        new_num_bolts = max(4, int(np.floor(max_num_bolts / 4.0) * 4))
        msgs.append(f"Number of bolts reduced {num_bolts} → {new_num_bolts} to prevent bolt hole overlap.")
        num_bolts = new_num_bolts

    # 5. Thickness check (must be positive)
    if thickness <= 0:
        thickness = 10.0
        msgs.append("Thickness was non-positive; set to 10.0 mm.")
        
    # 6. Clamp bolt diameter
    if bolt_d < 5:
        bolt_d = 5.0
        msgs.append("Bolt diameter increased to minimum 5.0 mm.")

    return (outer_d, inner_d, thickness, bolt_circle_d, bolt_d, num_bolts), msgs

# ===============================================================
# 3️⃣ Load Model and Scalers
# ===============================================================
@st.cache_resource
def load_resources_flange():
    try:
        # NOTE: Using compile=False as the model is for prediction only
        model = load_model(MODEL_PATH, compile=False)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        return model, preprocessor, scaler_y
    except Exception as e:
        # This is the exact error handler that catches the missing file
        st.error(f"❌ Failed to load model or scalers. Please check paths/run training: {e}")
        return None, None, None

model, preprocessor, scaler_y = load_resources_flange()

# ===============================================================
# 4️⃣ Title + Inputs
# ===============================================================
st.markdown("<h1 style='text-align:center;'>🧠 AI Physics-Informed Flange Parameter Enhancer + DXF Generator</h1>",
            unsafe_allow_html=True)
st.caption("Predict optimized flange geometry using physics + AI and export DXF (blue preview).")

st.markdown("### 📏 Initial Geometry Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    inner_d = st.number_input("Inner Diameter (mm)", 10.0, 500.0, 100.0, step=1.0)
    outer_d_init = st.number_input("Outer Diameter (mm)", 50.0, 800.0, 200.0, step=1.0)
    thickness_init = st.number_input("Flange Thickness (mm)", 5.0, 100.0, 25.0, step=0.1)
with col2:
    bolt_circle_d_init = st.number_input("Bolt Circle Diameter (mm)", 40.0, 750.0, 150.0, step=1.0)
    bolt_d_init = st.number_input("Bolt Diameter (mm)", 5.0, 50.0, 16.0, step=0.1)
    num_bolts_init = st.number_input("Number of Bolts", 4, 32, 8, step=4)
with col3:
    material = st.selectbox("Material", ["CarbonSteel", "Stainless", "Alloy", "CastIron"])
    gasket = st.selectbox("Gasket Type", ["SpiralWound", "RingJoint", "FlatMetal"])
    pressure_class = st.selectbox("Pressure Class", [150, 300, 600, 900])

st.markdown("### ⚙️ Operating Conditions")
colA, colB = st.columns(2)
with colA:
    internal_pressure = st.number_input("Internal Pressure (MPa)", 0.1, 150.0, 10.0, step=0.1)
    temperature = st.slider("Operating Temperature (°C)", 20, 500, 100)
with colB:
    axial_force = st.number_input("Axial Load (N)", 0.0, 5e5, 50000.0, step=100.0)
    bending_moment = st.number_input("Bending Moment (N·m)", 0.0, 5000.0, 500.0, step=10.0)

# ===============================================================
# 5️⃣ DXF Generator (API-style logic integrated)
# ===============================================================

def generate_flange_dxf(outer_diameter, inner_diameter, thickness, hole_diameter, num_holes, filename="flange_output.dxf"):
    """
    API-style DXF generator (uses ezdxf.new()).
    Returns (file_path, bytes_data)
    """
    # Convert to numeric types
    outer_diameter = float(outer_diameter)
    inner_diameter = float(inner_diameter)
    thickness = float(thickness)
    hole_diameter = float(hole_diameter)
    num_holes = int(num_holes)

    doc = ezdxf.new()
    msp = doc.modelspace()

    offset_x = 200
    offset_y = 250
    text_height = 10
    line_thickness = 0.35

    # Simple bolt circle radius heuristic
    bolt_circle_radius = (outer_diameter + inner_diameter) / 4.0
    if bolt_circle_radius <= inner_diameter / 2.0 + 1.0:
        bolt_circle_radius = (inner_diameter / 2.0) + (outer_diameter - inner_diameter) / 4.0

    angle_step = 360.0 / max(1, num_holes)
    hole_positions = [
        (
            bolt_circle_radius * np.cos(np.radians(i * angle_step)),
            bolt_circle_radius * np.sin(np.radians(i * angle_step)),
        )
        for i in range(num_holes)
    ]

    # FRONT VIEW (Top center)
    front_view_offset = (-200.0, -20.0)
    # Outer circle (color 1)
    msp.add_circle(center=front_view_offset, radius=outer_diameter / 2.0,
                   dxfattribs={"color": 1, "lineweight": line_thickness})
    # Inner circle (color 3)
    msp.add_circle(center=front_view_offset, radius=inner_diameter / 2.0,
                   dxfattribs={"color": 3, "lineweight": line_thickness})
    # holes
    for pos in hole_positions:
        msp.add_circle(center=(front_view_offset[0] + pos[0], front_view_offset[1] + pos[1]),
                       radius=hole_diameter / 2.0,
                       dxfattribs={"color": 5, "lineweight": line_thickness})
    msp.add_text("Front View", dxfattribs={"height": text_height}).dxf.insert = (
        front_view_offset[0] - 30, front_view_offset[1] - outer_diameter / 2.0 - 40
    )

    # SIDE VIEW (Right of Front View)
    side_view_offset = (offset_x, -10.0)
    rect_left = side_view_offset[0] - thickness / 2.0
    rect_right = side_view_offset[0] + thickness / 2.0
    rect_top = side_view_offset[1] + outer_diameter / 2.0
    rect_bottom = side_view_offset[1] - outer_diameter / 2.0

    # Center Bore in Side View (horizontal lines)
    center_y = side_view_offset[1]
    bore_left = side_view_offset[0] - thickness / 2.0
    bore_right = side_view_offset[0] + thickness / 2.0

    msp.add_line(
        start=(bore_left, center_y - inner_diameter / 2.0),
        end=(bore_right, center_y - inner_diameter / 2.0),
        dxfattribs={"color": 6}
    )
    msp.add_line(
        start=(bore_left, center_y + inner_diameter / 2.0),
        end=(bore_right, center_y + inner_diameter / 2.0),
        dxfattribs={"color": 6}
    )

    msp.add_lwpolyline(
        [
            (rect_left, rect_top),
            (rect_right, rect_top),
            (rect_right, rect_bottom),
            (rect_left, rect_bottom),
            (rect_left, rect_top)
        ],
        dxfattribs={"color": 7, "lineweight": line_thickness}
    )
    msp.add_text("Side View", dxfattribs={"height": text_height}).dxf.insert = (
        side_view_offset[0] - 60, rect_bottom - 40
    )

    # TOP VIEW (Below Front View)
    top_view_offset = (front_view_offset[0], front_view_offset[1] - outer_diameter - 50.0)

    rect_top_t = top_view_offset[1] + thickness / 2.0
    rect_bottom_t = top_view_offset[1] - thickness / 2.0
    rect_left_t = top_view_offset[0] - outer_diameter / 2.0
    rect_right_t = top_view_offset[0] + outer_diameter / 2.0

    # Center Bore in Top View (vertical lines)
    center_x = top_view_offset[0]
    msp.add_line(
        start=(center_x - inner_diameter / 2.0, rect_top_t),
        end=(center_x - inner_diameter / 2.0, rect_bottom_t),
        dxfattribs={"color": 6}
    )
    msp.add_line(
        start=(center_x + inner_diameter / 2.0, rect_top_t),
        end=(center_x + inner_diameter / 2.0, rect_bottom_t),
        dxfattribs={"color": 6}
    )

    # Draw rotated rectangle (thickness as height)
    msp.add_lwpolyline(
        [
            (rect_left_t, rect_top_t),
            (rect_right_t, rect_top_t),
            (rect_right_t, rect_bottom_t),
            (rect_left_t, rect_bottom_t),
            (rect_left_t, rect_top_t)
        ],
        dxfattribs={"color": 7, "lineweight": line_thickness}
    )

    # Add label
    msp.add_text("Top View", dxfattribs={"height": text_height}).dxf.insert = (
        top_view_offset[0] - 30, rect_bottom_t - 40
    )

    # Parameter Annotations
    base_x, base_y = -250.0, rect_bottom_t - 120.0
    spacing = 28.0
    annotations = [
        f"Outer Diameter: {outer_diameter:.2f} mm",
        f"Inner Diameter: {inner_diameter:.2f} mm",
        f"Flange Thickness: {thickness:.2f} mm",
        f"Hole Diameter: {hole_diameter:.2f} mm",
        f"Number of Holes: {num_holes}"
    ]
    for i, text in enumerate(annotations):
        msp.add_text(text, dxfattribs={"height": text_height}).dxf.insert = (base_x, base_y - i * spacing)

    # Save and return file
    file_name = filename
    file_path = os.path.join(OUTPUT_CONTENT_DIR, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    doc.saveas(file_path)
    with open(file_path, "rb") as f:
        data = f.read()
    return file_path, data


def generate_flange_dxf_from_params(params, inner_d_user, filename="flange_output.dxf"):
    """
    Wrapper to keep compatibility with the rest of the app:
    params: (outer_d, inner_d, thickness, bolt_circle_d, bolt_d, num_bolts)
    inner_d_user: the user's fixed inner diameter
    """
    # Use outer, thickness, bolt_d as hole diameter and num_bolts as number of holes for this simpler layout
    outer_d, _, thickness, bolt_circle_d, bolt_d, num_bolts = params
    # For the API-style generator we will treat bolt_d as hole_diameter and num_bolts as num_holes.
    return generate_flange_dxf(outer_d, inner_d_user, thickness, bolt_d, num_bolts, filename=filename)


# API-style glue (for external endpoint wiring if desired)
def create_file_1(outer, inner, thickness, holeDiameter, numHoles):
    """
    Accept numeric arguments, call the DXF generator, return file path
    """
    # Convert all values to float or int as needed
    outer = float(outer)
    inner = float(inner)
    thickness = float(thickness)
    hole_diameter = float(holeDiameter)
    num_holes = int(numHoles)

    # Call the DXF generation function
    file_path, _ = generate_flange_dxf(outer, inner, thickness, hole_diameter, num_holes, filename="flange_output_api.dxf")
    return file_path

def generate_file_1(request_json):
    """
    Simulates an API request handler that receives JSON payload
    request_json: dict-like with keys: outerDiameter, innerDiameter, thickness, holeDiameter, numHoles
    Returns bytes data for download (mimics send_file)
    """
    data = request_json
    outer = data.get('outerDiameter')
    inner = data.get('innerDiameter')
    thickness = data.get('thickness')
    holeDiameter = data.get('holeDiameter')
    numHoles = data.get('numHoles')

    generated_file_path = create_file_1(outer, inner, thickness, holeDiameter, numHoles)
    # Return raw bytes (in a real API you'd send the file)
    with open(generated_file_path, "rb") as f:
        file_bytes = f.read()
    return generated_file_path, file_bytes

# ===============================================================
# 6️⃣ DXF Preview Plot (force blue)
# ===============================================================
def plot_preview_flange(params, inner_d_user):
    temp_name = "_preview_blue.dxf"
    dxf_path, _ = generate_flange_dxf_from_params(params, inner_d_user, filename=temp_name)
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Force all entities blue
    for ent in msp:
        try:
            if hasattr(ent.dxf, "color"):
                ent.dxf.color = 5 # Blue
        except:
            pass

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, facecolor="white")
    ctx = RenderContext(doc)
    out = ez_mpl.MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp)

    # Make all text blue too
    for txt in ax.findobj(match=plt.Text):
        txt.set_color("blue")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig

# ===============================================================
# 7️⃣ Predict and DXF Generation (with validation & correction)
# ===============================================================
st.markdown("---")
if st.button("🔮 Predict Optimized Flange Geometry", use_container_width=True):
    
    # Store initial geometry for model input
    initial_geom = {
        "outer_d": outer_d_init, 
        "thickness": thickness_init, 
        "bolt_circle_d": bolt_circle_d_init, 
        "bolt_diameter": bolt_d_init, 
        "num_bolts": num_bolts_init
    }

    if model is not None:
        try:
            input_data = pd.DataFrame([{
                "inner_d": inner_d,
                "outer_d": outer_d_init,
                "thickness": thickness_init,
                "bolt_circle_d": bolt_circle_d_init,
                "bolt_diameter": bolt_d_init,
                "num_bolts": num_bolts_init,
                "material": material,
                "gasket": gasket,
                "internal_pressure": internal_pressure,
                "temperature": temperature,
                "axial_force": axial_force,
                "bending_moment": bending_moment
            }])
            
            # 7.1. Transform and Predict
            X_proc = preprocessor.transform(input_data)
            y_pred_scaled = model.predict(X_proc)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

            # Output variables: outer_d_opt, thickness_opt, bolt_circle_opt, bolt_diameter_opt, num_bolts_opt
            outer_d_pred, thickness_pred, bolt_circle_d_pred, bolt_d_pred, num_bolts_pred = y_pred

            # 7.2. Validation and Correction using user's fixed inner_d
            if not is_valid_flange(inner_d, outer_d_pred, thickness_pred, bolt_circle_d_pred, bolt_d_pred, num_bolts_pred):
                (outer_corr, inner_corr_fixed, thickness_corr, bolt_circle_corr, bolt_d_corr, num_bolts_corr), corrections = correct_flange_geometry(
                    inner_d, outer_d_pred, thickness_pred, bolt_circle_d_pred, bolt_d_pred, num_bolts_pred)
                for msg in corrections:
                    st.warning("⚠️ " + msg)
                
                # Update with corrected values (inner_d remains fixed from user input)
                outer_d_pred = outer_corr
                thickness_pred = thickness_corr
                bolt_circle_d_pred = bolt_circle_corr
                bolt_d_pred = bolt_d_corr
                num_bolts_pred = num_bolts_corr
            
            # Final validated/corrected parameters
            predicted_params = (outer_d_pred, inner_d, thickness_pred, bolt_circle_d_pred, bolt_d_pred, num_bolts_pred)
            st.session_state["predicted_params_flange"] = predicted_params
            st.session_state["inner_d_fixed"] = inner_d # Store fixed inner_d

            st.success("✅ Prediction & validation complete.")
            
            # Show table
            op, ip_fixed, tp, bcd, bd, nb = predicted_params
            
            comparison_data = {
                "Parameter": ["Outer Diameter (mm)", "Inner Diameter (mm)", "Thickness (mm)", "Bolt Circle D (mm)", "Bolt Diameter (mm)", "Number of Bolts"],
                "Initial Value": [round(outer_d_init, 2), round(inner_d, 2), round(thickness_init, 2), round(bolt_circle_d_init, 2), round(bolt_d_init, 2), int(round(num_bolts_init))],
                "Optimized Value": [round(op, 2), round(ip_fixed, 2), round(tp, 2), round(bcd, 2), round(bd, 2), int(round(nb))]
            }
            st.write(pd.DataFrame(comparison_data).set_index("Parameter"))

            # ===============================================================
            # ⚙️ Physics + AI Validation Dashboard
            # ===============================================================
            st.markdown("## ⚙️ Physics + AI Validation Dashboard")
            st.markdown("#### Real-time analytical insights from flange mechanics equations and AI prediction behavior.")

            colA, colB = st.columns(2)
            colC, colD = st.columns(2)

            # --- 1️⃣ Required Thickness vs Pressure ---
            pressure_range = np.linspace(max(0.01, internal_pressure) * 0.5, max(0.01, internal_pressure) * 2, 100)
            required_thickness = (pressure_range / 10.0) * (inner_d / 50.0) * (1.0 + 0.1 * (temperature / 100))
            base_thickness = thickness_pred
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=pressure_range, y=required_thickness,
                mode='lines', line=dict(color='darkblue', width=3),
                name='Required Thickness (Physics)'
            ))
            fig1.add_trace(go.Scatter(
                x=[internal_pressure], y=[base_thickness],
                mode='markers', marker=dict(color='red', size=10, symbol='star'),
                name='AI Optimized Thickness'
            ))
            fig1.update_layout(
                title="1️⃣ Required Thickness vs Internal Pressure",
                xaxis_title="Pressure (MPa)",
                yaxis_title="Required Thickness (mm)",
                template="plotly_white",
                height=360
            )
            colA.plotly_chart(fig1, use_container_width=True)

            # --- 2️⃣ Bolt Load Stress vs Bolt Diameter ---
            total_load = internal_pressure * (np.pi * (inner_d / 2.0)**2) + axial_force
            bolt_d_range = np.arange(max(1.0, bolt_d_init * 0.7), bolt_d_init * 1.3 + 1, 1)
            num_bolts_float = float(nb)

            stress_range = total_load / (num_bolts_float * np.pi * (bolt_d_range / 2.0)**2)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=bolt_d_range, y=stress_range,
                mode='lines', line=dict(color='steelblue', width=3),
                name='Bolt Stress Trend'
            ))
            fig2.add_trace(go.Scatter(
                x=[bd], y=[total_load / (num_bolts_float * np.pi * (bd / 2.0)**2)],
                mode='markers', marker=dict(color='red', size=10, symbol='circle'),
                name='AI Optimized Bolt Stress'
            ))
            fig2.update_layout(
                title="2️⃣ Bolt Stress vs Bolt Diameter",
                xaxis_title="Bolt Diameter (mm)",
                yaxis_title="Bolt Stress (N/mm²)",
                template="plotly_white",
                height=360
            )
            colB.plotly_chart(fig2, use_container_width=True)

            # --- 3️⃣ Flange Moment vs Gasket Type ---
            base_moment = internal_pressure * (np.pi * (bcd / 2.0)**2) * (bcd - ip_fixed)/4 + bending_moment
            
            gasket_factors = {"SpiralWound": 1.0, "RingJoint": 0.8, "FlatMetal": 1.2}
            gasket_moments = [base_moment * factor for factor in gasket_factors.values()]
            gasket_names = list(gasket_factors.keys())
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=gasket_names, y=gasket_moments,
                marker_color=['#1E90FF' if name != gasket else 'red' for name in gasket_names],
                name='Flange Moment'
            ))
            fig3.update_layout(
                title="3️⃣ Calculated Flange Moment vs Gasket Type",
                xaxis_title="Gasket Type",
                yaxis_title="Moment (N·m)",
                template="plotly_white",
                height=360
            )
            colC.plotly_chart(fig3, use_container_width=True)

            # --- 4️⃣ Temperature Derating Curve ---
            temp_range = np.linspace(20, 500, 100)
            derating = 1.0 - 0.001 * (temp_range - 20)
            derating[derating < 0.2] = 0.2
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=temp_range, y=derating * 100,
                mode='lines', line=dict(color='darkred', width=3),
                name='Strength Derating'
            ))
            fig4.add_trace(go.Scatter(
                x=[temperature], y=[(1.0 - 0.001 * (temperature - 20)) * 100],
                mode='markers', marker=dict(color='red', size=10, symbol='diamond'),
                name='Operating Point'
            ))
            fig4.update_layout(
                title="4️⃣ Flange Material Strength Derating vs Temperature",
                xaxis_title="Temperature (°C)",
                yaxis_title="Relative Strength (%)",
                template="plotly_white",
                height=360,
                yaxis=dict(range=[0, 105])
            )
            colD.plotly_chart(fig4, use_container_width=True)


        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("⚠️ Model not loaded properly. Check paths and ensure the training script ran correctly.")

# ===============================================================
# 8️⃣ DXF Visualization and Export
# ===============================================================
if "predicted_params_flange" in st.session_state and "inner_d_fixed" in st.session_state:
    final_params = st.session_state["predicted_params_flange"]
    inner_d_fixed = st.session_state["inner_d_fixed"]

    st.markdown("---")
    st.subheader("📐 DXF Visualization & Export")
    st.pyplot(plot_preview_flange(final_params, inner_d_fixed))

    if st.button("💾 Generate DXF File", use_container_width=True):
        # Use wrapper to create DXF compatible with this app flow
        fname = f"flange_{int(final_params[0])}_{int(final_params[1])}_{int(final_params[2])}_{int(final_params[5])}.dxf"
        fpath, data = generate_flange_dxf_from_params(final_params, inner_d_fixed, filename=fname)
        st.success(f"✅ DXF File Generated: {fname}")
        st.download_button(label="⬇️ Download DXF", data=data,
                           file_name=fname, mime="application/octet-stream")

# ===============================================================
# 9️⃣ Footer
# ===============================================================
st.markdown("---")
st.markdown("""
### 📘 About This App
This **AI-Physics-Informed Flange Enhancer** merges **ASME/pressure vessel mechanics** and **AI learning**
to predict optimal flange geometries (Outer D, Thickness, Bolt Circle, Bolt D, Bolt Count) considering:
- Internal Pressure & Operating Temperature  
- Material & Gasket Type  
- Axial Load & Bending Moment  

**Model:** Trained on 60k samples | Built with *TensorFlow + Scikit-learn + Streamlit + EZDXF*.
""")
