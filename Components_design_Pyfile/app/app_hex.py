"""
AI Physics-Informed Hex-Nut Parameter Enhancer + DXF Generator
Run:
    streamlit run app_physics_dxf_hexnut.py
"""

# ===============================================================
# 0️⃣ Imports
# ===============================================================
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import math
from tensorflow.keras.models import load_model
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import matplotlib as ez_mpl
from ezdxf.addons.drawing import RenderContext, Frontend

# ===============================================================
# 1️⃣ Configuration
# ===============================================================
st.set_page_config(
    page_title="AI Physics-Informed Hex-Nut Enhancer",
    layout="wide",
    page_icon="🔩",
    initial_sidebar_state="expanded"
)

# Simple styling matching flange app
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

# Paths — adjust if needed
BASE_DIR = r"C:\Users\Sanika\Documents\Project III\IEEE"
MODEL_PATH = os.path.join(BASE_DIR, "hexnut_enhancer_model.h5")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "hexnut_preprocessor.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "hexnut_scaler_y.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_hexnut")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CONTENT_DIR = os.path.join(OUTPUT_DIR, "content")
os.makedirs(OUTPUT_CONTENT_DIR, exist_ok=True)

# ===============================================================
# 2️⃣ Utility: Validation & Correction for hex nut geometry
# ===============================================================
def is_valid_hexnut(nominal_d, pitch, nut_thickness, width_af, chamfer, hole_depth):
    """Basic geometric sanity checks for a hex nut."""
    try:
        nominal_d = float(nominal_d)
        pitch = float(pitch)
        nut_thickness = float(nut_thickness)
        width_af = float(width_af)
        chamfer = float(chamfer)
        hole_depth = float(hole_depth)
    except:
        return False

    # Basic constraints:
    if nominal_d <= 0 or pitch <= 0 or nut_thickness <= 0 or width_af <= 0:
        return False
    # width across flats should be > nominal diameter (typical)
    if not (width_af > nominal_d * 1.1):
        return False
    # chamfer reasonable relative to nominal
    if chamfer > nominal_d * 0.3:
        return False
    # hole_depth approx <= nut_thickness * 1.2
    if hole_depth > nut_thickness * 1.5:
        return False
    return True

def correct_hexnut_geometry(nominal_d, pitch, nut_thickness, width_af, chamfer, hole_depth):
    """
    Attempt to correct common issues and report messages.
    Returns (corrected_tuple, messages)
    """
    msgs = []
    nominal_d = float(nominal_d)
    pitch = float(pitch)
    nut_thickness = float(nut_thickness)
    width_af = float(width_af)
    chamfer = float(chamfer)
    hole_depth = float(hole_depth)

    # Ensure positive
    if nominal_d <= 0:
        nominal_d = 8.0
        msgs.append("Nominal diameter set to default 8.0 mm.")
    if pitch <= 0:
        pitch = 1.25
        msgs.append("Thread pitch set to default 1.25 mm.")
    if nut_thickness <= 0:
        nut_thickness = max(0.6 * nominal_d, 3.0)
        msgs.append(f"Nut thickness set to {nut_thickness:.2f} mm.")

    # Width across flats: must be larger than nominal_d
    if width_af <= nominal_d * 1.05:
        new_w = nominal_d * 1.5
        msgs.append(f"Width A/F adjusted {width_af:.2f} → {new_w:.2f} (must be > nominal dia).")
        width_af = new_w

    # Chamfer clamp
    max_cham = nominal_d * 0.2
    if chamfer > max_cham:
        msgs.append(f"Chamfer reduced {chamfer:.2f} → {max_cham:.2f}.")
        chamfer = max_cham
    if chamfer <= 0:
        chamfer = max(0.05 * nominal_d, 0.5)
        msgs.append(f"Chamfer set to {chamfer:.2f} mm.")

    # Hole depth clamp
    if hole_depth <= 0 or hole_depth > nut_thickness * 1.5:
        new_h = nut_thickness * 0.95
        msgs.append(f"Hole depth adjusted to {new_h:.2f} mm.")
        hole_depth = new_h

    return (nominal_d, pitch, nut_thickness, width_af, chamfer, hole_depth), msgs

# ===============================================================
# 3️⃣ Load Model + Preprocessors (cached)
# ===============================================================
@st.cache_resource
def load_resources_hexnut():
    try:
        model = load_model(MODEL_PATH, compile=False)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        return model, preprocessor, scaler_y
    except Exception as e:
        st.error(f"❌ Failed to load hexnut model or scalers: {e}")
        return None, None, None

model, preprocessor, scaler_y = load_resources_hexnut()

# ===============================================================
# 4️⃣ UI Inputs
# ===============================================================
st.markdown("<h1>🧠 AI Physics-Informed Hex-Nut Parameter Enhancer + DXF Generator</h1>", unsafe_allow_html=True)
st.caption("Predict optimized hex-nut geometry and export DXF. Model trained on synthetic physics-informed data.")

st.markdown("### 📏 Initial Geometry Parameters")
col1, col2 = st.columns(2)
with col1:
    nominal_d = st.number_input("Nominal Diameter (mm)", 2.0, 60.0, 12.0, step=0.1)
    pitch_init = st.number_input("Thread Pitch (mm)", 0.5, 6.0, 1.75, step=0.01)
    nut_thickness_init = st.number_input("Nut Thickness (mm)", 1.0, 60.0, 10.0, step=0.1)
    width_af_init = st.number_input("Width Across Flats (mm)", 4.0, 120.0, 19.0, step=0.1)
with col2:
    chamfer_init = st.number_input("Chamfer Size (mm)", 0.0, 10.0, 0.8, step=0.1)
    hole_depth_init = st.number_input("Hole Depth (mm)", 0.5, 60.0, 9.0, step=0.1)
    material = st.selectbox("Material", ["CarbonSteel", "Stainless", "Alloy"])
    coating = st.selectbox("Coating", ["BlackOxide", "Zinc", "Phosphate"])
    
colA, colB = st.columns(2)
with colA:
    surface_finish = st.selectbox("Surface Finish", ["Standard", "Fine", "Precision"])
with colB:
    tolerance_level = st.selectbox("Tolerance Level", ["Loose", "Standard", "Tight"])

# ===============================================================
# 5️⃣ DXF generator (your provided logic adapted)
# ===============================================================
def generate_hex_nut_dxf_file(nominal_diameter, thread_pitch, nut_thickness, width_across_flats, chamfer_size, hole_depth, filename="hex_nut_output.dxf"):
    """
    Returns (file_path, bytes_data)
    Uses ezdxf to create Front / Side / Top and annotations.
    """
    # convert types
    nominal_diameter = float(nominal_diameter)
    thread_pitch = float(thread_pitch)
    nut_thickness = float(nut_thickness)
    width_across_flats = float(width_across_flats)
    chamfer_size = float(chamfer_size)
    hole_depth = float(hole_depth)

    doc = ezdxf.new()
    msp = doc.modelspace()

    offset_x = 0
    offset_y = 0
    text_height = 3
    line_thickness = 0.18

    # FRONT VIEW (HEX SHAPE)
    front_offset = ( -200.0, 0.0 )
    hex_radius = width_across_flats / (math.sqrt(3))
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.radians(angle_deg)
        x = front_offset[0] + hex_radius * math.cos(angle_rad)
        y = front_offset[1] + hex_radius * math.sin(angle_rad)
        points.append((x, y))
    points.append(points[0])

    for i in range(6):
        msp.add_line(points[i], points[i+1], dxfattribs={"color": 1, "lineweight": line_thickness})

    msp.add_circle(center=front_offset, radius=nominal_diameter/2, dxfattribs={"color": 3, "lineweight": line_thickness})

    # centerlines
    msp.add_line((front_offset[0] - hex_radius - 8, front_offset[1]), (front_offset[0] + hex_radius + 8, front_offset[1]),
                 dxfattribs={"color": 7, "lineweight": 0.09, "linetype": "CENTER"})
    msp.add_line((front_offset[0], front_offset[1] - hex_radius - 8), (front_offset[0], front_offset[1] + hex_radius + 8),
                 dxfattribs={"color": 7, "lineweight": 0.09, "linetype": "CENTER"})

    msp.add_text("FRONT VIEW", dxfattribs={"height": text_height}).dxf.insert = (front_offset[0] - 30, front_offset[1] - hex_radius - 18)

    # SIDE VIEW (right of front)
    side_offset = (front_offset[0] + 260.0, 0.0)
    half_width = width_across_flats / 2.0
    half_thickness = nut_thickness / 2.0
    hole_radius = nominal_diameter / 2.0

    # outer rectangle (section)
    msp.add_lwpolyline([
        (side_offset[0] - half_thickness, half_width),
        (side_offset[0] + half_thickness, half_width),
        (side_offset[0] + half_thickness, -half_width),
        (side_offset[0] - half_thickness, -half_width),
        (side_offset[0] - half_thickness, half_width)
    ], dxfattribs={"color":2, "lineweight": line_thickness})

    # center bore lines
    msp.add_line((side_offset[0] - half_thickness, hole_radius), (side_offset[0] + half_thickness, hole_radius),
                 dxfattribs={"color":3, "lineweight": line_thickness})
    msp.add_line((side_offset[0] - half_thickness, -hole_radius), (side_offset[0] + half_thickness, -hole_radius),
                 dxfattribs={"color":3, "lineweight": line_thickness})

    # chamfers
    msp.add_line((side_offset[0] - half_thickness, half_width), (side_offset[0] - half_thickness + chamfer_size, half_width - chamfer_size),
                 dxfattribs={"color":5, "lineweight":0.25})
    msp.add_line((side_offset[0] + half_thickness, half_width), (side_offset[0] + half_thickness - chamfer_size, half_width - chamfer_size),
                 dxfattribs={"color":5, "lineweight":0.25})
    msp.add_line((side_offset[0] - half_thickness, -half_width), (side_offset[0] - half_thickness + chamfer_size, -half_width + chamfer_size),
                 dxfattribs={"color":5, "lineweight":0.25})
    msp.add_line((side_offset[0] + half_thickness, -half_width), (side_offset[0] + half_thickness - chamfer_size, -half_width + chamfer_size),
                 dxfattribs={"color":5, "lineweight":0.25})

    msp.add_text("SIDE VIEW", dxfattribs={"height": text_height}).dxf.insert = (side_offset[0] - 40, -half_width - 14)

    # TOP VIEW (below front)
    top_offset = (front_offset[0], front_offset[1] - (hex_radius * 2.0) - 90.0)
    top_points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.radians(angle_deg)
        x = top_offset[0] + hex_radius * math.cos(angle_rad)
        y = top_offset[1] + hex_radius * math.sin(angle_rad)
        top_points.append((x, y))
    top_points.append(top_points[0])

    for i in range(6):
        msp.add_line(top_points[i], top_points[i+1], dxfattribs={"color": 1, "lineweight": line_thickness})

    msp.add_circle(center=top_offset, radius=nominal_diameter/2, dxfattribs={"color": 3, "lineweight": line_thickness})
    msp.add_text("TOP VIEW", dxfattribs={"height": text_height}).dxf.insert = (top_offset[0] - 25, top_offset[1] - hex_radius - 12)

    # Annotations below top view
    anno_x = front_offset[0]
    anno_y = top_offset[1] - hex_radius - 55
    spacing = 14
    notes = [
        f"Nominal Dia: {nominal_diameter:.2f} mm",
        f"Thread Pitch: {thread_pitch:.2f} mm",
        f"Nut Thickness: {nut_thickness:.2f} mm",
        f"Width A/F: {width_across_flats:.2f} mm",
        f"Chamfer: {chamfer_size:.2f} mm",
        f"Hole Depth: {hole_depth:.2f} mm"
    ]
    for i, n in enumerate(notes):
        msp.add_text(n, dxfattribs={"height": text_height}).dxf.insert = (anno_x, anno_y - i * spacing)

    # Save & return bytes
    file_path = os.path.join(OUTPUT_CONTENT_DIR, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    doc.saveas(file_path)
    with open(file_path, "rb") as f:
        data = f.read()
    return file_path, data

# wrapper used by app to fit pipeline
def generate_hex_nut_dxf_from_params(params, filename="hex_nut_output.dxf"):
    """
    params: predicted tuple (nominal_d, pitch, nut_thickness, width_af, chamfer, hole_depth)
    """
    nominal_d, pitch, nut_thickness, width_af, chamfer, hole_depth = params
    return generate_hex_nut_dxf_file(nominal_d, pitch, nut_thickness, width_af, chamfer, hole_depth, filename=filename)

# ===============================================================
# 6️⃣ DXF Preview (force blue)
# ===============================================================
def plot_preview_hexnut(pred_params):
    temp_name = "_preview_hex_blue.dxf"
    dxf_path, _ = generate_hex_nut_dxf_from_params(pred_params, filename=temp_name)
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Force all entities blue (color code 5)
    for ent in msp:
        try:
            if hasattr(ent.dxf, "color"):
                ent.dxf.color = 5
        except:
            pass

    fig = plt.figure(figsize=(7,6), facecolor="white")
    ax = fig.add_subplot(1,1,1, facecolor="white")
    ctx = RenderContext(doc)
    out = ez_mpl.MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp)

    # Make all text blue
    for txt in ax.findobj(match=plt.Text):
        txt.set_color("blue")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig

# ===============================================================
# 7️⃣ Predict & Generate (with validation)
# ===============================================================
st.markdown("---")
if st.button("🔮 Predict Optimized Hex-Nut Geometry", use_container_width=True):
    # Build input dataframe for model
    input_df = pd.DataFrame([{
        "nominal_d": nominal_d,
        "pitch": pitch_init,
        "nut_thickness": nut_thickness_init,
        "width_flats": width_af_init,
        "chamfer": chamfer_init,
        "hole_depth": hole_depth_init,
        "material": material,
        "coating": coating
    }])

    if model is None or preprocessor is None or scaler_y is None:
        st.warning("⚠️ Hex-nut model or preprocessors not loaded. Run training or check paths.")
    else:
        try:
            X_proc = preprocessor.transform(input_df)
            y_pred_scaled = model.predict(X_proc)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

            # Predicted outputs mapping (based on training targets)
            nut_thickness_pred, width_flats_pred, chamfer_pred, hole_depth_pred = y_pred

            # Pack predicted param set (nominal & pitch remain user inputs)
            predicted_params = (nominal_d, pitch_init, nut_thickness_pred, width_flats_pred, chamfer_pred, hole_depth_pred)

            # Validate predicted geometry
            if not is_valid_hexnut(*predicted_params):
                corrected, msgs = correct_hexnut_geometry(*predicted_params)
                for m in msgs:
                    st.warning("⚠️ " + m)
                predicted_params = tuple(corrected)

            # Save to session for preview/export
            st.session_state["predicted_params_hexnut"] = predicted_params

            st.success("✅ Prediction complete.")
            # Show comparison table
            init_vals = [nominal_d, pitch_init, nut_thickness_init, width_af_init, chamfer_init, hole_depth_init]
            opt_vals = [round(v,3) if isinstance(v,(int,float)) else v for v in predicted_params]
            comparison_data = {
                "Parameter": ["Nominal Dia (mm)", "Thread Pitch (mm)", "Nut Thickness (mm)", "Width A/F (mm)", "Chamfer (mm)", "Hole Depth (mm)"],
                "Initial Value": init_vals,
                "Optimized Value": opt_vals
            }
            st.write(pd.DataFrame(comparison_data).set_index("Parameter"))

            # Simple verification plots (physics-inspired)
            st.markdown("## ⚙️ Physics + AI Quick Insights")
            col1, col2 = st.columns(2)

            import plotly.graph_objects as go

            # ============================================================
            # Custom Engineering Plot Generator (Flexible)
            # ============================================================
            def engineering_bar_plot(title, x, y, x_label, y_label, std_min, std_max):

                fig = go.Figure()

                # 1) Physics bar graph  
                fig.add_trace(go.Bar(
                    x=x,
                    y=y,
                    name="Physics Relationship",
                    marker=dict(color="royalblue", opacity=0.85),
                ))

                # 2) Tolerance lines  
                fig.add_trace(go.Scatter(
                    x=x, y=std_min,
                    mode='lines',
                    name="Lower Tolerance",
                    line=dict(color="cyan", width=1.5, dash="dash")
                ))

                fig.add_trace(go.Scatter(
                    x=x, y=std_max,
                    mode='lines',
                    name="Upper Tolerance",
                    line=dict(color="cyan", width=1.5, dash="dash")
                ))

                fig.update_layout(
                    title=title,
                    paper_bgcolor="#0D0D0D",
                    plot_bgcolor="#111111",
                    font=dict(color="white"),
                    margin=dict(l=40, r=20, t=40, b=40),
                    hovermode="x unified"
                )

                fig.update_xaxes(title=x_label, gridcolor="#333")
                fig.update_yaxes(title=y_label, gridcolor="#333")

                return fig



            # ============================================================
            # Scientific Curve Plot (Different Style)
            # ============================================================
            def scientific_curve_plot(title, x, y, x_label, y_label, std_min, std_max):

                fig = go.Figure()

                # 1) Main smooth curve  
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="lines",
                    line=dict(width=3),
                    name="Physics Curve"
                ))

                # 2) Filled tolerance band  
                fig.add_trace(go.Scatter(
                    x=x, y=std_max,
                    mode="lines",
                    line=dict(width=0),
                    name="Upper Limit",
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=x, y=std_min,
                    mode="lines",
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor="rgba(0,255,0,0.25)",
                    name="Tolerance Band"
                ))

                fig.update_layout(
                    title=title,
                    paper_bgcolor="#0D0D0D",
                    plot_bgcolor="#111111",
                    font=dict(color="white"),
                    margin=dict(l=40, r=20, t=40, b=40),
                    hovermode="x unified"
                )

                fig.update_xaxes(title=x_label, gridcolor="#333")
                fig.update_yaxes(title=y_label, gridcolor="#333")

                return fig



            # ============================================================
            # A) Nominal Diameter → Nut Thickness (Engineering Bar Style)
            # ============================================================
            d_vals = np.linspace(5, 60, 80)

            # Physics m = 0.8*d
            m_vals = 0.8 * d_vals

            # 10% ISO tolerance
            std_min = 0.72 * d_vals
            std_max = 0.88 * d_vals

            fig1 = engineering_bar_plot(
                title="Nominal Diameter → Nut Thickness (Physics Based)",
                x=d_vals,
                y=m_vals,
                x_label="nominal_d (mm)",
                y_label="nut_thickness (mm)",
                std_min=std_min,
                std_max=std_max
            )

            col1.plotly_chart(fig1, use_container_width=True)



            # ============================================================
            # B) Pitch → Width Across Flats (Scientific Curve Style)
            # ============================================================
            p_vals = np.linspace(0.5, 4.0, 60)

            # Convert pitch to approximate diameter  
            d_equiv = p_vals * 10

            # Width across flats S = 1.5 * d
            s_vals = 1.5 * d_equiv

            # ±3% tolerance
            std_min2 = 1.455 * d_equiv
            std_max2 = 1.545 * d_equiv

            fig2 = scientific_curve_plot(
                title="Thread Pitch → Width Across Flats (Physics Based)",
                x=p_vals,
                y=s_vals,
                x_label="pitch (mm)",
                y_label="width_across_flats (mm)",
                std_min=std_min2,
                std_max=std_max2
            )

            col2.plotly_chart(fig2, use_container_width=True)








        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ===============================================================
# 8️⃣ DXF Visualization & Export
# ===============================================================
if "predicted_params_hexnut" in st.session_state:
    st.markdown("---")
    st.subheader("📐 DXF Visualization & Export")
    final_params = st.session_state["predicted_params_hexnut"]
    st.pyplot(plot_preview_hexnut(final_params))

    if st.button("💾 Generate HEX-NUT DXF", use_container_width=True):
        fname = f"hexnut_{int(final_params[0])}_{int(final_params[2])}.dxf"
        fpath, data = generate_hex_nut_dxf_from_params(final_params, filename=fname)
        st.success(f"✅ DXF File Generated: {fname}")
        st.download_button(label="⬇️ Download DXF", data=data, file_name=fname, mime="application/octet-stream")

# ===============================================================
# 9️⃣ Footer
# ===============================================================
st.markdown("---")
st.markdown("""
### 📘 About This App
This **AI-Physics-Informed Hex-Nut Enhancer** predicts optimized hex-nut geometry (thickness, width across flats, chamfer, hole depth)
based on nominal diameter, pitch, material and coating, combining physics-inspired heuristics and data-driven learning.
Model: Trained separately with TensorFlow + Scikit-learn + EZDXF for DXF export.
""")
