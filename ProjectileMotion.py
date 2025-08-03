import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile

def app(data=None):
    st.title("Inclined Plane Simulator (Animated)")

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

    # Forces & kinematics
    f_parallel = mass * g * np.sin(theta_rad)
    f_normal = mass * g * np.cos(theta_rad)
    f_friction = mu * f_normal
    f_net = f_parallel - f_friction
    acceleration = f_net / mass

    # Final velocity: v² = 2aL
    final_velocity = np.sqrt(2 * acceleration * length) if acceleration > 0 else 0
    time = final_velocity / acceleration if acceleration > 0 else 0

    st.markdown("### Results")
    st.markdown(f"- **Acceleration:** `{acceleration:.2f} m/s²`")
    st.markdown(f"- **Final Velocity:** `{final_velocity:.2f} m/s`")
    st.markdown(f"- **Time to reach bottom:** `{time:.2f} s`")

    # Kinematics
    t = np.linspace(0, time, 120) if time > 0 else np.array([0])
    s = 0.5 * acceleration * t**2
    s = np.clip(s, 0, length)  # <-- Clamp block position to ramp length

    # Ramp geometry: always top-left (start) to bottom-right (end)
    x0, y0 = 0, length * np.sin(theta_rad)  # top of ramp
    x1, y1 = length * np.cos(theta_rad), 0  # bottom of ramp

    x_block = x0 + s * np.cos(theta_rad)
    y_block = y0 - s * np.sin(theta_rad)

    # ---- Animation Section ----
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([x0, x1], [y0, y1], 'k-', lw=4, label="Ramp")
    ax.set_xlim(-0.2 * length, x1 + 0.2 * length)
    ax.set_ylim(-0.2 * length, y0 + 0.3 * length)
    ax.set_xlabel("Horizontal (m)")
    ax.set_ylabel("Vertical (m)")
    ax.set_title("Block Sliding Down an Inclined Plane")
    block, = ax.plot([], [], 'ro', markersize=14, label="Block")

    # Draw ground
    ax.plot([x0, x1, x1 + 0.2 * length], [y1, y1, y1], 'brown', lw=2)

    def init():
        block.set_data([], [])
        return block,

    def animate(i):
        idx = min(i, len(x_block) - 1)
        # Don't let the block go past the bottom of the ramp
        if s[idx] >= length:
            block.set_data([x1], [y1])
        else:
            block.set_data([x_block[idx]], [y_block[idx]])
        return block,

    ani = FuncAnimation(fig, animate, frames=len(x_block), init_func=init, blit=True, interval=25)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer='pillow')
    plt.close(fig)

    st.image(tmpfile.name, caption="Inclined Plane Animation", use_container_width=True)

    # Plots for position/velocity vs time (optional)
    with st.expander("View Position & Velocity vs Time Graphs"):
        fig2, ax2 = plt.subplots()
        ax2.plot(t, s, label="Position (m)")
        ax2.plot(t, acceleration * t, label="Velocity (m/s)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Motion on Inclined Plane")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

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

