"""
AI Physics-Informed Pulley Parameter Enhancer + DXF Generator
Run:
    streamlit run app_pulley.py
"""

# ===============================================================
# 0️⃣ Import Libraries
# ===============================================================
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import math
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
    page_title="AI Physics-Informed Pulley Enhancer",
    layout="wide",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)

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

# Paths - adjust if you saved models elsewhere
BASE_DIR = r"C:\Users\Sanika\Documents\Project III\IEEE"
MODEL_PATH = os.path.join(BASE_DIR, "pulley_model.h5")
SCALER_X_PATH = os.path.join(BASE_DIR, "pulley_scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "pulley_scaler_y.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_pulley")
OUTPUT_CONTENT_DIR = os.path.join(OUTPUT_DIR, "content")
os.makedirs(OUTPUT_CONTENT_DIR, exist_ok=True)

# ===============================================================
# 2️⃣ Utility: Validation & Correction functions
# ===============================================================
def is_valid_pulley(outer_d, bore_d, width, groove_angle, side_width, bottom_width):
    """Simple geometry checks for a pulley."""
    try:
        outer_d = float(outer_d); bore_d = float(bore_d); width = float(width)
        groove_angle = float(groove_angle); side_width = float(side_width); bottom_width = float(bottom_width)
    except:
        return False

    if outer_d <= 0 or bore_d <= 0 or width <= 0: return False
    if not (bore_d < outer_d * 0.6): return False  # bore should be reasonably smaller
    if not (20 <= groove_angle <= 60): return False  # typical V grooves ~28-42, allow larger bounds
    if bottom_width <= 0 or side_width <= 0: return False
    # groove must fit inside outer radius
    if bottom_width/2.0 + side_width > outer_d/2.0:
        return False
    return True

def correct_pulley_geometry(outer_d, bore_d, width, groove_angle, side_width, bottom_width):
    """Attempt simple corrections and return (params,msgs)."""
    msgs = []
    outer_d = float(outer_d); bore_d = float(bore_d); width = float(width)
    groove_angle = float(groove_angle); side_width = float(side_width); bottom_width = float(bottom_width)

    # Ensure positives
    if outer_d <= 0:
        outer_d = 100.0
        msgs.append("Outer diameter was non-positive; set to 100 mm.")
    if bore_d <= 0:
        bore_d = max(8.0, outer_d * 0.1)
        msgs.append(f"Bore diameter set to {bore_d:.1f} mm.")
    if width <= 0:
        width = 30.0
        msgs.append("Width set to 30 mm.")
    # Clamp groove angle to usable range
    if groove_angle < 20:
        msgs.append(f"Groove angle increased {groove_angle:.1f} → 28.0° (min practical).")
        groove_angle = 28.0
    if groove_angle > 60:
        msgs.append(f"Groove angle reduced {groove_angle:.1f} → 42.0° (max practical).")
        groove_angle = 42.0

    # Ensure groove fits radially
    groove_depth = side_width * math.sin(math.radians(groove_angle/2))
    if (bottom_width/2.0 + groove_depth) >= (outer_d/2.0 - 1.0):
        # increase outer_d a bit
        needed = (bottom_width/2.0 + groove_depth) - (outer_d/2.0) + 5.0
        new_outer = outer_d + max(needed, 5.0)
        msgs.append(f"Outer diameter increased {outer_d:.2f} → {new_outer:.2f} to fit groove depth.")
        outer_d = new_outer

    # Ensure bore reasonable
    if bore_d >= outer_d * 0.6:
        new_bore = outer_d * 0.4
        msgs.append(f"Bore diameter reduced {bore_d:.2f} → {new_bore:.2f} to keep rim thickness.")
        bore_d = new_bore

    # Bottom width not exceed side widths
    if bottom_width > side_width * 2.5:
        new_bottom = side_width * 1.5
        msgs.append(f"Bottom groove width reduced {bottom_width:.2f} → {new_bottom:.2f}.")
        bottom_width = new_bottom

    return (outer_d, bore_d, width, groove_angle, side_width, bottom_width), msgs

# ===============================================================
# 3️⃣ Load Model & Scalers (safe)
# ===============================================================
@st.cache_resource
def load_pulley_resources():
    model = None; scaler_X = None; scaler_y = None
    try:
        # load model without compiling to avoid deserialization issues
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Failed to load pulley model: {e}")
    try:
        scaler_X = joblib.load(SCALER_X_PATH)
    except Exception as e:
        st.warning(f"⚠️ Could not load input scaler: {e}")
    try:
        scaler_y = joblib.load(SCALER_Y_PATH)
    except Exception as e:
        st.warning(f"⚠️ Could not load output scaler: {e}")
        scaler_y = None
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_pulley_resources()

# ===============================================================
# 4️⃣ Title + Inputs
# ===============================================================
col1, col2 = st.columns(2)
with col1:
    outer_d_init = st.number_input("Outer Diameter (mm)", 40.0, 1000.0, 200.0, step=1.0)
    bore_d_init = st.number_input("Bore Diameter (mm)", 6.0, 500.0, 40.0, step=1.0)
    width_init = st.number_input("Total Width (mm)", 5.0, 300.0, 40.0, step=0.1)
with col2:
    groove_angle_init = st.number_input("Groove Angle (° included)", 20.0, 60.0, 34.0, step=0.1)
    groove_side_width_init = st.number_input("Groove Side Width (mm)", 1.0, 40.0, 8.0, step=0.1)
    bottom_groove_width_init = st.number_input("Bottom Groove Width (mm)", 0.5, 30.0, 4.0, step=0.1)

colA, colB = st.columns(2)
with colA:
    belt_tension = st.number_input("Estimated Belt Tension (N)", 0.0, 5000.0, 500.0, step=10.0)
    speed_rpm = st.number_input("Speed (RPM)", 0.0, 20000.0, 1200.0, step=10.0)
with colB:
    material = st.selectbox("Material", ["Steel", "Aluminium", "CastIron"])
    safety_factor = st.slider("Safety Factor", 1.0, 3.0, 1.5, step=0.1)

# ===============================================================
# 5️⃣ DXF Generator (Front, Side, Top views)
# ===============================================================
def generate_pulley_dxf(outer_diameter, bore_diameter, total_width,
                        groove_angle, groove_side_width, bottom_groove_width,
                        filename="pulley_output.dxf"):

    import ezdxf
    import math
    import os

    # -------------------------------
    # Convert and sanitize inputs
    # -------------------------------
    outer_diameter = float(outer_diameter)
    bore_diameter = float(bore_diameter)
    total_width = float(total_width)
    groove_angle = float(groove_angle)
    groove_side_width = float(groove_side_width)
    bottom_groove_width = float(bottom_groove_width)

    # Visual attributes
    text_height = 6

    outer_r = outer_diameter / 2
    bore_r = bore_diameter / 2
    half_w = total_width / 2
    spacing = outer_diameter + 120

    # -------------------------------
    # Groove geometry
    # -------------------------------
    groove_half_angle_rad = math.radians(groove_angle / 2)
    groove_depth = groove_side_width * math.sin(groove_half_angle_rad)
    groove_top_half = (
        bottom_groove_width/2 +
        groove_depth / max(1e-6, math.tan(groove_half_angle_rad))
    )

    # -------------------------------
    # Create DXF
    # -------------------------------
    doc = ezdxf.new()
    msp = doc.modelspace()

    # ============================================
    # 1️⃣ FRONT VIEW (CENTER)
    # ============================================
    fx, fy = 0, 0

    msp.add_circle((fx, fy), outer_r, dxfattribs={"color":1})
    msp.add_circle((fx, fy), bore_r, dxfattribs={"color":3})

    # Centerlines
    msp.add_line((fx - outer_r - 5, fy), (fx + outer_r + 5, fy),
                 dxfattribs={"color":7})
    msp.add_line((fx, fy - outer_r - 5), (fx, fy + outer_r + 5),
                 dxfattribs={"color":7})

    txt = msp.add_text("FRONT VIEW", dxfattribs={"height": text_height})
    txt.dxf.insert = (fx - 25, fy - outer_r - 25)

    # ============================================
    # 2️⃣ SIDE VIEW (RIGHT OF FRONT)
    # ============================================
    sx, sy = fx + spacing, fy

    msp.add_lwpolyline([
        (sx - half_w, outer_r),
        (sx + half_w, outer_r),
        (sx + half_w, -outer_r),
        (sx - half_w, -outer_r),
        (sx - half_w, outer_r),
    ], dxfattribs={"color":2})

    # Bore slot lines
    msp.add_line((sx - half_w, bore_r), (sx + half_w, bore_r), dxfattribs={"color":3})
    msp.add_line((sx - half_w, -bore_r), (sx + half_w, -bore_r), dxfattribs={"color":3})

    # Grooves in side view
    # Upper groove
    msp.add_line((sx - groove_top_half, outer_r),
                 (sx - bottom_groove_width/2, outer_r - groove_depth),
                 dxfattribs={"color":5})
    msp.add_line((sx + groove_top_half, outer_r),
                 (sx + bottom_groove_width/2, outer_r - groove_depth),
                 dxfattribs={"color":5})
    msp.add_line((sx - bottom_groove_width/2, outer_r - groove_depth),
                 (sx + bottom_groove_width/2, outer_r - groove_depth),
                 dxfattribs={"color":5})

    # Lower groove
    msp.add_line((sx - groove_top_half, -outer_r),
                 (sx - bottom_groove_width/2, -(outer_r - groove_depth)),
                 dxfattribs={"color":5})
    msp.add_line((sx + groove_top_half, -outer_r),
                 (sx + bottom_groove_width/2, -(outer_r - groove_depth)),
                 dxfattribs={"color":5})
    msp.add_line((sx - bottom_groove_width/2, -(outer_r - groove_depth)),
                 (sx + bottom_groove_width/2, -(outer_r - groove_depth)),
                 dxfattribs={"color":5})

    txt = msp.add_text("SIDE VIEW (SECTION)", dxfattribs={"height": text_height})
    txt.dxf.insert = (sx - 40, -outer_r - 25)

    # ============================================
    # 3️⃣ TOP VIEW (BELOW FRONT)
    # ============================================
    tx, ty = fx, fy - spacing

    msp.add_lwpolyline([
        (tx - outer_r, ty + half_w),
        (tx - outer_r, ty - half_w),
        (tx + outer_r, ty - half_w),
        (tx + outer_r, ty + half_w),
        (tx - outer_r, ty + half_w),
    ], dxfattribs={"color":2})

    # Bore slot
    msp.add_line((tx - bore_r, ty + half_w), (tx - bore_r, ty - half_w), dxfattribs={"color":3})
    msp.add_line((tx + bore_r, ty + half_w), (tx + bore_r, ty - half_w), dxfattribs={"color":3})

    # Left groove
    msp.add_line((tx - outer_r, ty + groove_top_half),
                 (tx - (outer_r - groove_depth), ty + bottom_groove_width/2),
                 dxfattribs={"color":5})
    msp.add_line((tx - outer_r, ty - groove_top_half),
                 (tx - (outer_r - groove_depth), ty - bottom_groove_width/2),
                 dxfattribs={"color":5})
    msp.add_line((tx - (outer_r - groove_depth), ty + bottom_groove_width/2),
                 (tx - (outer_r - groove_depth), ty - bottom_groove_width/2),
                 dxfattribs={"color":5})

    # Right groove
    msp.add_line((tx + outer_r, ty + groove_top_half),
                 (tx + (outer_r - groove_depth), ty + bottom_groove_width/2),
                 dxfattribs={"color":5})
    msp.add_line((tx + outer_r, ty - groove_top_half),
                 (tx + (outer_r - groove_depth), ty - bottom_groove_width/2),
                 dxfattribs={"color":5})
    msp.add_line((tx + (outer_r - groove_depth), ty + bottom_groove_width/2),
                 (tx + (outer_r - groove_depth), ty - bottom_groove_width/2),
                 dxfattribs={"color":5})

    txt = msp.add_text("TOP VIEW", dxfattribs={"height": text_height})
    txt.dxf.insert = (tx - 20, ty - half_w - 25)

    # ============================================
    # 4️⃣ ANNOTATIONS (BELOW TOP VIEW)
    # ============================================
    anno_x = tx - 90
    anno_y = ty - half_w - 80

    annotations = [
        f"Outer Diameter: {outer_diameter:.2f} mm",
        f"Bore Diameter : {bore_diameter:.2f} mm",
        f"Width: {total_width:.2f} mm",
        f"Groove Angle: {groove_angle:.2f}°",
        f"Groove Side Width: {groove_side_width:.2f} mm",
        f"Bottom Groove Width: {bottom_groove_width:.2f} mm"
    ]

    for i, text in enumerate(annotations):
        txt = msp.add_text(text, dxfattribs={"height": text_height})
        txt.dxf.insert = (anno_x, anno_y - i*16)

    # ============================================
    # SAVE
    # ============================================
    out_dir = "OUTPUT_DXF"
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, filename)

    doc.saveas(file_path)
    with open(file_path, "rb") as f:
        data = f.read()

    return file_path, data



def generate_pulley_dxf_from_params(params, filename="pulley_output.dxf"):
    outer_d, bore_d, width, groove_angle, side_width, bottom_width = params
    return generate_pulley_dxf(outer_d, bore_d, width, groove_angle, side_width, bottom_width, filename=filename)

# ===============================================================
# 6️⃣ DXF Preview Plot (force colors for clarity)
# ===============================================================
def plot_preview_pulley(params):
    temp_name = "_preview_pulley.dxf"
    dxf_path, _ = generate_pulley_dxf_from_params(params, filename=temp_name)
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Force some color for clarity
    for ent in msp:
        try:
            if hasattr(ent.dxf, "color"):
                ent.dxf.color = 1
        except:
            pass

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, facecolor="white")
    ctx = RenderContext(doc)
    out = ez_mpl.MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp)

    # set text color
    for txt in ax.findobj(match=plt.Text):
        txt.set_color("black")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig

# ===============================================================
# 7️⃣ Predict & DXF Generation (with validation & correction)
# ===============================================================
st.markdown("---")
if st.button("🔮 Predict Optimized Pulley Geometry", use_container_width=True):
    # prepare input for model (features order must match training)
    input_df = pd.DataFrame([{
        "outer_d": outer_d_init,
        "bore_d": bore_d_init,
        "width": width_init,
        "groove_angle": groove_angle_init,
        "groove_side_width": groove_side_width_init,
        "bottom_groove_width": bottom_groove_width_init,
        # optionally include physics features like belt_tension, speed, material, safety_factor if model expects them
    }])

    if model is None or scaler_X is None or scaler_y is None:
        st.warning("⚠️ Model or scalers not loaded correctly. Check training output files and paths.")
    else:
        try:
            # scale inputs with scaler_X (was used in training)
            X_proc = scaler_X.transform(input_df.values)
            y_pred_scaled = model.predict(X_proc)
            # inverse scale outputs
            y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

            # outputs were: outer_d_opt, width_opt, groove_angle_opt, bottom_width_opt  (in your training)
            # But training for pulley earlier used 4 outputs; your UI expects full geometry - we'll map as best-effort:
            # We'll construct full params using predicted items and keep bore/side widths from input or predicted where applicable.
            outer_pred = float(y_pred[0])
            width_pred = float(y_pred[1])
            groove_angle_pred = float(y_pred[2])
            bottom_width_pred = float(y_pred[3])
            # Keep bore and side width from user (model training didn't optimize them in the given sample),
            bore_pred = float(bore_d_init)
            side_width_pred = float(groove_side_width_init)

            params = (outer_pred, bore_pred, width_pred, groove_angle_pred, side_width_pred, bottom_width_pred)

            # Validate / correct
            if not is_valid_pulley(*params):
                (params_corr, corrections) = correct_pulley_geometry(*params)
                for msg in corrections:
                    st.warning("⚠️ " + msg)
                params = params_corr

            st.session_state["predicted_params_pulley"] = params
            st.success("✅ Prediction & validation complete.")
            op, bp, wp, ga, sw, bw = params

            st.write(pd.DataFrame({
                "Parameter": ["Outer Diameter (mm)", "Bore Diameter (mm)", "Width (mm)", "Groove Angle (°)", "Groove Side Width (mm)", "Bottom Groove Width (mm)"],
                "Optimized Value": [round(op,2), round(bp,2), round(wp,2), round(ga,2), round(sw,2), round(bw,2)]
            }).set_index("Parameter"))

            # Physics + AI Dashboard
            st.markdown("## ⚙️ Physics + AI Validation Dashboard")
            col1, col2 = st.columns(2)

            # Life/trend / simple physics: estimated belt contact circumference vs groove bottom width
            contact_circ = np.pi * op
            contact_area = contact_circ * bw
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=["Contact Area (approx)"], y=[contact_area], marker_color='steelblue'))
            fig1.update_layout(title="Estimated Belt Contact Area (approx)", template="plotly_white", height=300)
            col1.plotly_chart(fig1, use_container_width=True)

            # Groove depth vs side width relation (sanity)
            depth = sw * math.sin(math.radians(ga/2.0))
            fig2 = go.Figure()
            fig2.add_trace(go.Indicator(
                mode="gauge+number",
                value=depth,
                title={"text":"Groove Depth (mm approx)"},
                gauge={"axis": {"range": [0, max(10.0, depth*3)]}}
            ))
            col2.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ===============================================================
# 8️⃣ DXF Visualization & Export
# ===============================================================
if "predicted_params_pulley" in st.session_state:
    final_params = st.session_state["predicted_params_pulley"]

    st.markdown("---")
    st.subheader("📐 DXF Visualization & Export (Front / Side / Top)")

    # Preview
    st.pyplot(plot_preview_pulley(final_params))

    if st.button("💾 Generate DXF File", use_container_width=True):
        fname = f"pulley_{int(final_params[0])}_{int(final_params[1])}_{int(final_params[2])}.dxf"
        fpath, data = generate_pulley_dxf_from_params(final_params, filename=fname)
        st.success(f"✅ DXF File Generated: {fname}")
        st.download_button(label="⬇️ Download DXF", data=data, file_name=fname, mime="application/octet-stream")

# ===============================================================
# 9️⃣ Footer
# ===============================================================
st.markdown("---")
st.markdown("""
### 📘 About This App
This **AI-Physics-Informed Pulley Enhancer** blends simple physics heuristics with a neural model to suggest refined pulley geometry,
renders Front / Side / Top orthographic views (DXF), and exports DXF for CAD use.

**Notes**
- Ensure your `pulley_model.h5`, `pulley_scaler_X.pkl` and `pulley_scaler_y.pkl` are saved in the `BASE_DIR` path.
- Model is loaded with `compile=False` for safe deserialization.
""")
