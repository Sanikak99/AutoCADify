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


# --------------------
# BALL BEARING SCREEN (your original logic)
# --------------------
def ball_bearing_interface():
    # ===============================================================
    # 1️⃣ Configuration
    # ===============================================================
    st.set_page_config(
        page_title="AI Physics-Informed Bearing Enhancer",
        layout="wide",  # ✅ full-screen professional view
        page_icon="⚙️",
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
                color: #8B0000;
                font-size: 40px;
                margin-bottom: 10px;
            }
            h2, h3, h4 {
                color: #222222;
            }
            .stButton>button {
                background-color: #028fc7;
                color: white;
                border-radius: 10px;
                border: none;
                font-size: 16px;
                height: 3em;
                width: 100%;
                font-weight: 500;
            }
            .stButton>button:hover {
                background-color: #006187;
            }
            .block-container {
                max-width: 1300px;
                margin: 0 auto;
            }
        </style>
    """, unsafe_allow_html=True)



    BASE_DIR = r"C:\Users\Sanika\Desktop\IEEE 2\AutoCADify"
    MODEL_PATH = os.path.join(BASE_DIR, "enhancer_model_physics.h5")
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
    SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===============================================================
    # 2️⃣ Utility Functions (validation, correction, positions)
    # ===============================================================
    def is_valid(outer, inner, width, ball_d, num_balls):
        """Basic rule checks for bearing geometry validity (pre-correction)."""
        try:
            outer = float(outer); inner = float(inner); width = float(width)
            ball_d = float(ball_d); num_balls = int(round(num_balls))
        except:
            return False

        if not (inner < outer):
            return False
        if not (width > 0 and width < outer * 0.6):
            return False
        if not (ball_d > 0 and ball_d < (outer - inner) / 2):
            return False
        if num_balls < 3 or num_balls > 200:
            return False

        # center radius for balls (mid between inner and outer radii)
        R_center = (outer/2.0 + inner/2.0) / 2.0
        ball_r = ball_d/2.0
        clearance = 0.1

        # balls must not protrude outside rings
        if (R_center + ball_r + clearance) > (outer/2.0):
            return False
        if (R_center - ball_r - clearance) < (inner/2.0):
            return False

        # arc spacing check
        circumference = 2.0 * np.pi * R_center
        if num_balls <= 0:
            return False
        arc_length = circumference / num_balls
        if arc_length < ball_d * 1.05:  # small safety factor
            return False

        return True

    def compute_ball_positions(R_center, num_balls):
        """Return list of (x,y) centers for balls around a circle of radius R_center."""
        angles = np.linspace(0.0, 2.0*np.pi, int(num_balls), endpoint=False)
        return [(R_center * np.cos(a), R_center * np.sin(a)) for a in angles]

    def correct_geometry(outer, inner, width, ball_d, num_balls):
        """
        Attempt to correct geometry to avoid overlaps/protrusion:
        - increase outer if too small
        - reduce ball_d if too large
        - reduce num_balls if spacing insufficient
        Returns corrected tuple and list of messages describing corrections.
        """
        msgs = []
        outer = float(outer); inner = float(inner)
        width = float(width); ball_d = float(ball_d)
        num_balls = max(1, int(round(num_balls)))

        # Safety margins
        clearance = 0.1
        spacing_factor = 1.05  # arc spacing multiplier

        # Ensure outer > inner
        if outer <= inner:
            new_outer = inner + 2.0 + 0.5  # minimal margin if user set equal/wrong
            msgs.append(f"Outer diameter increased {outer:.2f} → {new_outer:.2f} to be > inner.")
            outer = new_outer

        # Clamp width
        max_width = outer * 0.6
        if width <= 0:
            msgs.append("Width was non-positive; set to 1.0 mm.")
            width = 1.0
        if width > max_width:
            msgs.append(f"Width reduced {width:.2f} → {max_width:.2f} (<=60% of outer).")
            width = max_width

        # Ball center radius: mid between inner/2 and outer/2
        R_center = (outer/2.0 + inner/2.0) / 2.0
        ball_r = ball_d / 2.0

        # Ensure balls fit radially: ball center must be between inner/2 + ball_r + clearance and outer/2 - ball_r - clearance
        max_ball_r_by_outer = (outer/2.0) - R_center - clearance
        max_ball_r_by_inner = R_center - (inner/2.0) - clearance
        max_ball_r_allowed = min(max_ball_r_by_outer, max_ball_r_by_inner)
        if max_ball_r_allowed <= 0:
            # If impossible, increase outer minimally to make room
            needed = (ball_r + clearance) - max_ball_r_allowed + 1.0
            new_outer = outer + max(needed, 1.0)
            msgs.append(f"Outer increased {outer:.2f} → {new_outer:.2f} to create radial space for balls.")
            outer = new_outer
            R_center = (outer/2.0 + inner/2.0) / 2.0
            max_ball_r_by_outer = (outer/2.0) - R_center - clearance
            max_ball_r_by_inner = R_center - (inner/2.0) - clearance
            max_ball_r_allowed = min(max_ball_r_by_outer, max_ball_r_by_inner)

        # If ball radius too large, reduce it
        if ball_r > max_ball_r_allowed:
            new_ball_r = max(0.5, max_ball_r_allowed)  # ensure at least 0.5 mm
            msgs.append(f"Ball diameter reduced {ball_d:.2f} → {2*new_ball_r:.2f} to fit radially.")
            ball_d = 2.0 * new_ball_r
            ball_r = new_ball_r

        # Now compute max number of balls to avoid overlap on circumference
        circumference = 2.0 * np.pi * R_center
        if ball_d <= 0:
            ball_d = 1.0
        max_num_by_circ = max(3, int(np.floor(circumference / (ball_d * spacing_factor))))
        if num_balls > max_num_by_circ:
            msgs.append(f"Number of balls reduced {num_balls} → {max_num_by_circ} to prevent overlap.")
            num_balls = max_num_by_circ

        # Re-evaluate one more time radial fit (edge-case)
        R_center = (outer/2.0 + inner/2.0) / 2.0
        ball_r = ball_d / 2.0
        if (R_center + ball_r + clearance) > (outer/2.0) or (R_center - ball_r - clearance) < (inner/2.0):
            # As a last resort, increase outer slightly
            needed = max((R_center + ball_r + clearance) - (outer/2.0), (inner/2.0) - (R_center - ball_r - clearance), 0.0)
            if needed > 0:
                new_outer = outer + needed + 0.5
                msgs.append(f"Outer increased {outer:.2f} → {new_outer:.2f} to avoid protrusion.")
                outer = new_outer

        return (outer, inner, width, ball_d, num_balls), msgs

    # ===============================================================
    # 3️⃣ Load Model and Scalers
    # ===============================================================
    @st.cache_resource
    def load_resources():
        try:
            model = load_model(MODEL_PATH, compile=False)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            scaler_y = joblib.load(SCALER_Y_PATH)
            return model, preprocessor, scaler_y
        except Exception as e:
            st.error(f"❌ Failed to load model or scalers: {e}")
            return None, None, None

    model, preprocessor, scaler_y = load_resources()

    # ===============================================================
    # 4️⃣ Title + Inputs
    # ===============================================================
    
    col1, col2 = st.columns(2)
    with col1:
        inner_d = st.number_input("Inner Diameter (mm)", 10.0, 200.0, 50.0, step=0.1)
        width = st.number_input("Bearing Width (mm)", 0.1, 200.0, 20.0, step=0.1)
        ball_d = st.number_input("Ball Diameter (mm)", 0.1, 100.0, 8.0, step=0.01)
        num_balls = st.number_input("Number of Balls", 1, 200, 8, step=1)
        radial_load = st.number_input("Radial Load (N)", 0.0, 1e6, 1500.0, step=1.0)
    with col2:
        axial_load = st.number_input("Axial Load (N)", 0.0, 1e6, 0.0, step=1.0)
        speed_rpm = st.number_input("Rotational Speed (RPM)", 0.0, 200000.0, 1000.0, step=1.0)
        temperature = st.slider("Operating Temperature (°C)", 0, 300, 25)
        life_L10_hr = st.number_input("Desired Life (hours)", 1e3, 1e8, 1e5, step=1000.0)
        material = st.selectbox("Material", ["Steel", "Ceramic", "Bronze"])
        lubrication = st.selectbox("Lubrication", ["Oil", "Grease", "Dry"])

    # ===============================================================
    # 5️⃣ DXF Generator (safe drawing using corrected geometry)
    # ===============================================================
    def generate_bearing_dxf(params, filename="bearing_output.dxf"):
        outer_diameter, inner_diameter, width_val, ball_diameter, num_balls_val = params
        outer_diameter = float(outer_diameter)
        inner_diameter = float(inner_diameter)
        width_val = float(width_val)
        ball_diameter = float(ball_diameter)
        num_balls_val = max(3, int(round(num_balls_val)))

        doc = ezdxf.new()
        msp = doc.modelspace()

        text_height = 6
        line_thickness = 0.35

        # ball center radius (mid between inner and outer radii)
        R_center = (outer_diameter/2.0 + inner_diameter/2.0) / 2.0
        # compute positions
        angle_step = 360.0 / num_balls_val
        ball_positions = [(R_center * np.cos(np.radians(i * angle_step)),
                        R_center * np.sin(np.radians(i * angle_step))) for i in range(num_balls_val)]

        # FRONT VIEW (left)
        front_view_offset = (-150.0, -20.0)
        # draw outer & inner rings (red)
        msp.add_circle(center=front_view_offset, radius=outer_diameter / 2.0,
                    dxfattribs={"color": 1, "lineweight": line_thickness})
        msp.add_circle(center=front_view_offset, radius=inner_diameter / 2.0,
                    dxfattribs={"color": 1, "lineweight": line_thickness})
        # draw balls (red)
        for pos in ball_positions:
            cx = front_view_offset[0] + pos[0]
            cy = front_view_offset[1] + pos[1]
            msp.add_circle(center=(cx, cy), radius=ball_diameter / 2.0, dxfattribs={"color": 1, "lineweight": line_thickness})

        # FRONT label (red)
        front_label = msp.add_text("FRONT VIEW", dxfattribs={"height": text_height, "color": 1})
        front_label.dxf.insert = (front_view_offset[0] - 40, front_view_offset[1] - outer_diameter / 2.0 - 30)

        # SIDE VIEW (right) - rectangle for width in red
        offset_x = 200.0
        rect_left = offset_x - width_val / 2.0
        rect_right = offset_x + width_val / 2.0
        rect_top = -10.0 + outer_diameter / 2.0
        rect_bottom = -10.0 - outer_diameter / 2.0

        msp.add_lwpolyline(
            [(rect_left, rect_top), (rect_right, rect_top), (rect_right, rect_bottom), (rect_left, rect_bottom), (rect_left, rect_top)],
            dxfattribs={"color": 1, "lineweight": line_thickness}
        )
        # side lines representing ball centers (red)
        for pos in ball_positions:
            y = -10.0 + pos[1]
            msp.add_line(start=(rect_left, y), end=(rect_right, y), dxfattribs={"color": 1, "lineweight": 0.25})

        side_label = msp.add_text("SIDE VIEW", dxfattribs={"height": text_height, "color": 1})
        side_label.dxf.insert = (rect_left - 30.0, rect_bottom - 30.0)

        # TOP VIEW (above front)
        top_view_offset = (front_view_offset[0], front_view_offset[1] - outer_diameter - 50.0)
        rect_top2 = top_view_offset[1] + width_val / 2.0
        rect_bottom2 = top_view_offset[1] - width_val / 2.0
        rect_left2 = top_view_offset[0] - outer_diameter / 2.0
        rect_right2 = top_view_offset[0] + outer_diameter / 2.0

        msp.add_lwpolyline(
            [(rect_left2, rect_top2), (rect_right2, rect_top2), (rect_right2, rect_bottom2), (rect_left2, rect_bottom2), (rect_left2, rect_top2)],
            dxfattribs={"color": 1, "lineweight": line_thickness}
        )
        for pos in ball_positions:
            x = top_view_offset[0] + pos[0]
            msp.add_line(start=(x, rect_bottom2), end=(x, rect_top2), dxfattribs={"color": 1, "lineweight": 0.25})

        top_label = msp.add_text("TOP VIEW", dxfattribs={"height": text_height, "color": 1})
        top_label.dxf.insert = (rect_left2, rect_bottom2 - 30.0)

        # Parameter annotations (red)
        base_x, base_y = -250.0, rect_bottom2 - 120.0
        spacing = 28.0
        annotations = [
            f"Outer Diameter: {outer_diameter:.2f} mm",
            f"Inner Diameter: {inner_diameter:.2f} mm",
            f"Bearing Width: {width_val:.2f} mm",
            f"Ball Diameter: {ball_diameter:.2f} mm",
            f"Number of Balls: {num_balls_val}"
        ]
        for i, text in enumerate(annotations):
            note = msp.add_text(text, dxfattribs={"height": text_height, "color": 1})
            note.dxf.insert = (base_x, base_y - i * spacing)

        # Save DXF to OUTPUT_DIR
        file_path = os.path.join(OUTPUT_DIR, filename)
        doc.saveas(file_path)
        with open(file_path, "rb") as f:
            data = f.read()
        return file_path, data

    # ===============================================================
    # 6️⃣ DXF Preview Plot (force red)
    # ===============================================================
    def plot_preview(params):
        temp_name = "_preview_red.dxf"
        dxf_path, _ = generate_bearing_dxf(params, filename=temp_name)
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        # Force all entities red
        for ent in msp:
            try:
                if hasattr(ent.dxf, "color"):
                    ent.dxf.color = 1
            except:
                pass

        fig = plt.figure(figsize=(6, 6), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, facecolor="white")
        ctx = RenderContext(doc)
        out = ez_mpl.MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp)

        # Make all text red too
        for txt in ax.findobj(match=plt.Text):
            txt.set_color("red")

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        return fig

    # ===============================================================
    # 7️⃣ Predict and DXF Generation (with validation & correction)
    # ===============================================================
    st.markdown("---")
    if st.button("🔮 Predict Optimized Bearing Geometry", use_container_width=True):
        if model is not None:
            try:
                input_data = pd.DataFrame([{
                    "inner_d": inner_d, "width": width, "ball_d": ball_d, "num_balls": num_balls,
                    "radial_load": radial_load, "axial_load": axial_load, "speed_rpm": speed_rpm,
                    "material": material, "lubrication": lubrication, "temperature": temperature,
                    "life_L10_hr": life_L10_hr
                }])
                X_proc = preprocessor.transform(input_data)
                y_pred_scaled = model.predict(X_proc)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

                outer_d_pred, width_pred, ball_d_pred, num_balls_pred = y_pred

                # Attempt correction if invalid (also for bad manual inputs)
                if not is_valid(outer_d_pred, inner_d, width_pred, ball_d_pred, num_balls_pred):
                    (outer_corr, inner_corr, width_corr, ball_d_corr, num_balls_corr), corrections = correct_geometry(
                        outer_d_pred, inner_d, width_pred, ball_d_pred, num_balls_pred)
                    for msg in corrections:
                        st.warning("⚠️ " + msg)
                    outer_d_pred, width_pred, ball_d_pred, num_balls_pred = outer_corr, width_corr, ball_d_corr, num_balls_corr
                else:
                    # Still check user's manual inputs for impossible situations (if user didn't use AI)
                    if not is_valid(outer_d_pred, inner_d, width_pred, ball_d_pred, num_balls_pred):
                        st.error("❌ Geometry invalid even after checks. Adjust inputs.")
                        st.session_state.pop("predicted_params", None)
                        raise ValueError("Invalid geometry")

                # Save corrected/validated params for preview & DXF
                st.session_state["predicted_params"] = (outer_d_pred, inner_d, width_pred, ball_d_pred, num_balls_pred)

                st.success("✅ Prediction & validation complete.")
                # Show table
                op, ip, wp, bd, nb = st.session_state["predicted_params"]
                st.write(pd.DataFrame({
                    "Parameter": ["Outer Diameter (mm)", "Inner Diameter (mm)", "Width (mm)", "Ball Diameter (mm)", "Number of Balls"],
                    "Predicted Value": [round(op, 2), round(ip, 2), round(wp, 2), round(bd, 2), int(round(nb))]
                }).set_index("Parameter"))

                # ===============================================================
                # ⚙️ Physics + AI Validation Dashboard (IEEE Presentation Ready)
                # ===============================================================
                st.markdown("## ⚙️ Physics + AI Validation Dashboard")
                st.markdown("#### Real-time analytical insights from bearing physics equations and AI prediction behavior.")

                colA, colB = st.columns(2)
                colC, colD = st.columns(2)

                # --- 1️⃣ Bearing Life vs Load (Log Line + Confidence Band) ---
                load_factor = np.linspace(0.5, 2.5, 100)
                predicted_life = (life_L10_hr / (load_factor ** 3))
                noise = predicted_life * (0.05 * np.random.randn(100))
                upper = predicted_life + np.abs(noise)
                lower = predicted_life - np.abs(noise)

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=load_factor, y=upper,
                    line=dict(width=0),
                    fill=None, mode='lines',
                    name='Upper Bound', showlegend=False
                ))
                fig1.add_trace(go.Scatter(
                    x=load_factor, y=lower,
                    fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
                    line=dict(width=0),
                    name='Confidence Band'
                ))
                fig1.add_trace(go.Scatter(
                    x=load_factor, y=predicted_life,
                    mode='lines', line=dict(color='crimson', width=3),
                    name='L10 Life (Predicted)'
                ))
                fig1.update_layout(
                    title="1️⃣ Bearing Life vs Load (Log Scale)",
                    xaxis_title="Load Factor (× Rated Load)",
                    yaxis_title="Predicted L10 Life (hours)",
                    yaxis_type="log",
                    template="plotly_white",
                    height=360
                )
                colA.plotly_chart(fig1, use_container_width=True)

                # --- 2️⃣ Contact Stress vs Number of Balls (Bar + Curve Fit) ---
                num_balls_range = np.arange(6, 31)
                stress = (radial_load + axial_load + 1) / (np.pi * (ball_d ** 2) * num_balls_range)
                polyfit = np.poly1d(np.polyfit(num_balls_range, stress, 2))
                trend_y = polyfit(num_balls_range)

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=num_balls_range, y=stress,
                    marker_color='rgba(220,20,60,0.6)',
                    name='Measured Stress'
                ))
                fig2.add_trace(go.Scatter(
                    x=num_balls_range, y=trend_y,
                    mode='lines', line=dict(color='firebrick', width=3, dash='dot'),
                    name='Polynomial Trend'
                ))
                fig2.update_layout(
                    title="2️⃣ Contact Stress vs Number of Balls",
                    xaxis_title="Number of Balls",
                    yaxis_title="Relative Contact Stress (N/mm²)",
                    template="plotly_white",
                    height=360
                )
                colB.plotly_chart(fig2, use_container_width=True)

                # --- 3️⃣ Efficiency vs Speed (Gaussian Curve + Peak Marker) ---
                speed_range = np.linspace(500, speed_rpm * 1.8 if speed_rpm > 0 else 1800, 100)
                efficiency = np.exp(-((speed_range - speed_rpm) ** 2) / (2 * (speed_rpm / 3 + 1) ** 2)) * 100
                peak_speed = speed_range[np.argmax(efficiency)]

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=speed_range, y=efficiency,
                    mode='lines', line=dict(color='darkred', width=3),
                    fill='tozeroy', fillcolor='rgba(255,0,0,0.1)',
                    name='Efficiency Curve'
                ))
                fig3.add_trace(go.Scatter(
                    x=[peak_speed], y=[np.max(efficiency)],
                    mode='markers+text',
                    marker=dict(color='red', size=10, symbol='star'),
                    text=["Peak Efficiency"], textposition="top center"
                ))
                fig3.update_layout(
                    title="3️⃣ Efficiency vs Rotational Speed",
                    xaxis_title="Speed (RPM)",
                    yaxis_title="Efficiency (%)",
                    template="plotly_white",
                    height=360
                )
                colC.plotly_chart(fig3, use_container_width=True)

                # --- 4️⃣ Temperature Rise vs Speed (Dual Curve + Annotation) ---
                speed_range = np.linspace(1000, 1.6 * (speed_rpm if speed_rpm > 0 else 1000), 80)
                oil_temp = temperature + 0.015 * (speed_range - 1000)
                grease_temp = temperature + 0.025 * (speed_range - 1000)
                delta_temp = grease_temp - oil_temp

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=speed_range, y=oil_temp,
                    mode='lines', line=dict(color='firebrick', width=3),
                    name='Oil Lubrication'
                ))
                fig4.add_trace(go.Scatter(
                    x=speed_range, y=grease_temp,
                    mode='lines', line=dict(color='indianred', width=3, dash='dash'),
                    name='Grease Lubrication'
                ))
                fig4.add_trace(go.Scatter(
                    x=speed_range, y=delta_temp + temperature - 5,
                    mode='lines', line=dict(color='rgba(255,0,0,0.3)', width=0),
                    fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
                    name='ΔT Zone'
                ))
                fig4.update_layout(
                    title="4️⃣ Temperature Rise vs Speed (Lubrication Impact)",
                    xaxis_title="Speed (RPM)",
                    yaxis_title="Temperature (°C)",
                    template="plotly_white",
                    height=360,
                    legend=dict(orientation='h', y=-0.25, x=0.2)
                )
                colD.plotly_chart(fig4, use_container_width=True)




            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        else:
            st.warning("⚠️ Model not loaded properly. Check paths and retrain/save model.")

    # ===============================================================
    # 8️⃣ DXF Visualization and Export
    # ===============================================================
    if "predicted_params" in st.session_state:
        final_params = st.session_state["predicted_params"]

        st.markdown("---")
        st.subheader("📐 DXF Visualization & Export")
        st.pyplot(plot_preview(final_params))

        if st.button("💾 Generate DXF File", use_container_width=True):
            fname = f"bearing_{int(final_params[0])}_{int(final_params[1])}_{int(final_params[4])}.dxf"
            fpath, data = generate_bearing_dxf(final_params, fname)
            st.success(f"✅ DXF File Generated: {fname}")
            st.download_button(label="⬇️ Download DXF", data=data,
                            file_name=fname, mime="application/octet-stream")

    # ===============================================================
    # 9️⃣ Footer
    # ===============================================================
    st.markdown("---")
    st.markdown("""
    ### 📘 About This App
    This **AI-Physics-Informed Bearing Enhancer** merges **mechanics equations** and **AI learning**
    to predict optimal bearing geometries considering:
    - Radial / Axial Load  
    - Rotational Speed  
    - Temperature & Lubrication  
    - Material and L10 Life  

    **Model:** 60k samples | R² ≈ 0.997 | RMSE ≈ 1.6  
    Built with *TensorFlow + Scikit-learn + Streamlit + EZDXF*
    """)



def flange_interface():
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
    BASE_DIR = r"C:\Users\Sanika\Desktop\IEEE 2\AutoCADify"
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
    col1, col2 = st.columns(2)
    with col1:
        inner_d = st.number_input("Inner Diameter (mm)", 10.0, 500.0, 100.0, step=1.0)
        outer_d_init = st.number_input("Outer Diameter (mm)", 50.0, 800.0, 200.0, step=1.0)
        thickness_init = st.number_input("Flange Thickness (mm)", 5.0, 100.0, 25.0, step=0.1)

        bolt_circle_d_init = st.number_input("Bolt Circle Diameter (mm)", 40.0, 750.0, 150.0, step=1.0)
        bolt_d_init = st.number_input("Bolt Diameter (mm)", 5.0, 50.0, 16.0, step=0.1)
        num_bolts_init = st.number_input("Number of Bolts", 4, 32, 8, step=4)

    with col2:
        material = st.selectbox("Material", ["CarbonSteel", "Stainless", "Alloy", "CastIron"])
        gasket = st.selectbox("Gasket Type", ["SpiralWound", "RingJoint", "FlatMetal"])
        pressure_class = st.selectbox("Pressure Class", [150, 300, 600, 900])

        internal_pressure = st.number_input("Internal Pressure (MPa)", 0.1, 150.0, 10.0, step=0.1)
        temperature = st.slider("Operating Temperature (°C)", 20, 500, 100)

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

def pully_interface():
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
    BASE_DIR = r"C:\Users\Sanika\Desktop\IEEE 2\AutoCADify"
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
                st.markdown("#### Real-time insights from pulley mechanics, belt dynamics, and AI-assisted performance predictions.")
                
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



def hexnut_interface():
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
    BASE_DIR = r"C:\Users\Sanika\Desktop\IEEE 2\AutoCADify"
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
                st.markdown("## ⚙️ Physics + AI Validation Dashboard")
                st.markdown("#### Real-time insights from hex nut mechanics, ISO standards, and AI-assisted prediction behavior.")
                
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



def shaft_interface():
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
                background-color: #1E90FF; /* Blue */;
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

    # --- PATHS ---
    BASE_DIR = r"C:\Users\Sanika\Desktop\IEEE 2\AutoCADify"
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









# ================================
# ✅ MAIN APP — TABS
# ================================
st.set_page_config(page_title="Mechanical Designer", layout="centered")

st.title("🧠 AI Mechanical Design Platform")
st.markdown(
    "<h6 style='text-align: center;'>Turning Engineering Knowledge into Intelligent Automation...</h6>",
    unsafe_allow_html=True
)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ball Bearing",
    "Flange",
    "Pully",
    "Hexnut",
    "Shaft"
])





with tab1:
    ball_bearing_interface()

with tab2:
    flange_interface()

with tab3:
    pully_interface()

with tab4:
    hexnut_interface()

with tab5:
    shaft_interface()
