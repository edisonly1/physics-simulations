import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os

def app(data=None):
    st.title("Animated Projectile Motion Simulator")

    # Inputs
    if data:
        v0 = float(data.get("initial_velocity", 20))
        angle_deg = float(data.get("angle", 45))
        h0 = float(data.get("height", 0))
        st.markdown(f"**Initial Velocity:** {v0} m/s  \n**Angle:** {angle_deg}°  \n**Height:** {h0} m")
    else:
        v0 = st.slider("Initial Velocity (m/s)", 0.0, 50.0, 25.0, step=0.1)
        angle_deg = st.slider("Launch Angle (degrees)", 0.0, 90.0, 45.0, step=0.1)
        h0 = st.slider("Initial Height (m)", 0.0, 10.0, 0.0, step=0.1)

    # Physics
    g = 9.8
    theta = np.radians(angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t_flight = (vy + np.sqrt(vy**2 + 2 * g * h0)) / g
    t_vals = np.linspace(0, t_flight, 120)
    x_vals = vx * t_vals
    y_vals = h0 + vy * t_vals - 0.5 * g * t_vals**2

    # --- Animation ---
    fig, ax = plt.subplots()
    ax.set_xlim(0, np.max(x_vals)*1.05)
    ax.set_ylim(0, max(np.max(y_vals)*1.05, 1))
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Projectile Path (Animated)")

    line, = ax.plot([], [], 'b-', lw=2)
    point, = ax.plot([], [], 'ro', markersize=8)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def animate(i):
        line.set_data(x_vals[:i+1], y_vals[:i+1])
        point.set_data(x_vals[i], y_vals[i])
        return line, point

    ani = FuncAnimation(fig, animate, frames=len(t_vals), init_func=init, blit=True, interval=20)

    # Save to a temporary file
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer='pillow')

    plt.close(fig)  # Prevents double display in Streamlit

    # Show the GIF in Streamlit
    st.image(tmpfile.name, caption="Projectile Animation", use_column_width=True)

    # Remove temp file after use
    # os.remove(tmpfile.name)  # Uncomment for local cleanup; Streamlit Cloud cleans up temp automatically

    # Results (as before)
    st.markdown("### Results")
    st.markdown(f"- **Time of Flight:** `{t_flight:.2f}` seconds")
    st.markdown(f"- **Horizontal Range:** `{x_vals[-1]:.2f}` meters")


    with st.expander("View Calculations and Formulas"):
        st.markdown(r"""
        **Equations Used:**

        - Horizontal velocity:  $v_x = v_0 \cos(\theta)$  
        - Vertical velocity:    $v_y = v_0 \sin(\theta)$  
        - Time of flight:       $t = \frac{v_y + \sqrt{v_y^2 + 2gh}}{g}$  
        - Horizontal distance:  $x(t) = v_x \cdot t$  
        - Vertical position:    $y(t) = h + v_y t - \frac{1}{2} g t^2$  
        """)

