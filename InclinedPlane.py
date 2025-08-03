import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app(data=None):
    st.title("Inclined Plane Simulator")

    g = 10.0  # gravity

    use_ai = data is not None

    if use_ai:
        st.markdown("#### Using AI-extracted values:")
        angle = float(data.get("angle", 30))
        mass = float(data.get("mass", 1))
        mu = float(data.get("friction", 0))
        length = float(data.get("length", 5))  # length of ramp
    else:
        mass = st.slider("Mass (kg)", 0.1, 10.0, 1.0)
        angle = st.slider("Incline Angle (°)", 0.0, 90.0, 30.0)
        mu = st.slider("Friction Coefficient (μ)", 0.0, 1.0, 0.0)
        length = st.slider("Ramp Length (m)", 0.1, 20.0, 5.0)

    theta_rad = np.radians(angle)

    # Forces
    f_parallel = mass * g * np.sin(theta_rad)
    f_normal = mass * g * np.cos(theta_rad)
    f_friction = mu * f_normal
    f_net = f_parallel - f_friction
    acceleration = f_net / mass

    # Final velocity: v² = 2aL
    final_velocity = np.sqrt(2 * acceleration * length) if acceleration > 0 else 0
    time = final_velocity / acceleration if acceleration > 0 else 0

    # Results
    st.markdown("### Results")
    st.markdown(f"- **Acceleration:** `{acceleration:.2f} m/s²`")
    st.markdown(f"- **Final Velocity:** `{final_velocity:.2f} m/s`")
    st.markdown(f"- **Time to reach bottom:** `{time:.2f} s`")

    # Optional plots
    t = np.linspace(0, time, 300)
    v = acceleration * t
    s = 0.5 * acceleration * t**2

    fig, ax = plt.subplots()
    ax.plot(t, s, label="Position (m)")
    ax.plot(t, v, label="Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Motion on Inclined Plane")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    with st.expander("View Calculations and Formulas"):
        st.markdown(r"""
        **Equations Used:**

        - $F_{\parallel} = mg\sin(\theta)$  
        - $F_N = mg\cos(\theta)$  
        - $f_k = \mu mg\cos(\theta)$  
        - $a = \frac{F_{\parallel} - f_k}{m}$  
        - $v_f = \sqrt{2aL}$  
        - $t = \frac{v_f}{a}$
        """)
