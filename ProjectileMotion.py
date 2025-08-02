import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app(velocity=None, angle=None, height=0):
    st.title("Projectile Motion Simulator")

    # Use AI-parsed values if available, otherwise use defaults
    v0 = st.slider("Initial Velocity (m/s)", 0.0, 50.0, float(velocity) if velocity is not None else 25.0, step=0.1)
    angle_deg = st.slider("Launch Angle (degrees)", 0.0, 90.0, float(angle) if angle is not None else 45.0, step=0.1)
    h0 = st.slider("Initial Height (m)", 0.0, 10.0, float(height) if height is not None else 0.0, step=0.1)

    # Physics calculations
    g = 9.8
    theta = np.radians(angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    # Time of flight
    t_flight = (vy + np.sqrt(vy**2 + 2 * g * h0)) / g
    t = np.linspace(0, t_flight, 300)

    x = vx * t
    y = h0 + vy * t - 0.5 * g * t**2

    range_x = x[-1]

    # Display results
    st.markdown("###Results")
    st.markdown(f"- **Time of Flight:** `{t_flight:.2f}` seconds")
    st.markdown(f"- **Horizontal Range:** `{range_x:.2f}` meters")

    with st.expander("View Calculations and Formulas"):
        st.markdown(r"""
        **Equations Used:**

        - Horizontal velocity:  $v_x = v_0 \cos(\theta)$  
        - Vertical velocity:    $v_y = v_0 \sin(\theta)$  
        - Time of flight:       $t = \frac{v_y + \sqrt{v_y^2 + 2gh}}{g}$  
        - Horizontal distance:  $x(t) = v_x \cdot t$  
        - Vertical position:    $y(t) = h + v_y t - \frac{1}{2} g t^2$  
        """)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Projectile Path")
    ax.grid(True)
    st.pyplot(fig)
