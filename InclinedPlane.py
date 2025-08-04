import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
from InclineFBD import draw_incline_fbd

def app(data=None):
    st.title("Inclined Plane Simulator (Animated)")

    g = 10.0  # gravity

    use_ai = data is not None

    # Set defaults in case some values are missing from AI extraction
    angle = 30
    mass = 1
    mu = 0
    length = 5
    motion_type = ""
    question_type = ""

    if use_ai:
        st.markdown("#### Using AI-extracted values:")

        # Angle and mass
        try:
            angle = float(data.get("angle", 30))
        except Exception:
            angle = 30

        try:
            mass = float(data.get("mass", 1))
        except Exception:
            mass = 1

        # Friction: try coefficient, then type (frictionless, etc.)
        mu = 0.0
        fc = data.get("friction_coefficient")
        ft = str(data.get("friction_type", "")).lower()
        if fc is not None:
            try:
                mu = float(fc)
            except Exception:
                mu = 0.0
        elif "frictionless" in ft:
            mu = 0.0
        elif "static" in ft or "kinetic" in ft:
            mu = float(fc) if fc is not None else 0.2  # fallback

        # Use 'distance' as ramp length if present, else default
        try:
            length = float(data.get("length", 5) or 5)
        except Exception:
            length = 10

        motion_type = str(data.get("motion_type", "")).lower()
        question_type = str(data.get("question_type", "")).lower()
        direction_raw = str(data.get("direction", "down")).lower()
        if "up" in direction_raw:
            direction = "up"
        else:
            direction = "down"


        # Let user override ramp length if AI did not supply it
        if not data.get("distance"):
            length = st.slider("Ramp Length (m)", 0.1, 20.0, 5.0)
    else:
        mass = st.slider("Mass (kg)", 0.1, 10.0, 1.0)
        angle = st.slider("Incline Angle (°)", 0.0, 90.0, 30.0)
        mu = st.slider("Friction Coefficient (μ)", 0.0, 1.0, 0.0)
        length = st.slider("Ramp Length (m)", 0.1, 20.0, 5.0)
        direction = st.selectbox("Direction", ["down", "up"]).lower()

    theta_rad = np.radians(angle)

    # Forces & kinematics
    f_parallel = mass * g * np.sin(theta_rad)
    f_normal = mass * g * np.cos(theta_rad)
    f_friction = mu * f_normal
    # Determine net force and acceleration depending on direction
    if "up" in direction:
        # Friction and gravity both act to slow the block
        f_net = -f_parallel - f_friction
    else:
        # Block is going down: gravity helps, friction opposes
        f_net = f_parallel - f_friction
    acceleration = f_net / mass

    # Kinematics: assume starting from rest unless otherwise specified (or add support for initial_velocity!)
    # For up-the-ramp problems (e.g., "how far until stopping"), the logic may need v0 > 0.
    v0 = float(data.get("initial_velocity", 0)) if use_ai else 0.0

    if "up" in direction:
        # Block starts at bottom with v0 up the ramp, stops at distance d.
        # v^2 = v0^2 + 2ad, but final v = 0, so: d = -v0^2 / (2a)
        # a must be negative for the block to stop
        if acceleration >= 0:
            st.warning("Block does not stop (acceleration ≥ 0 going up). Check inputs.")
            return
        distance_to_stop = -v0 ** 2 / (2 * acceleration)
        if distance_to_stop > length:
            st.info(f"The block would go off the ramp after {length:.2f} m, before stopping.")
            stop_dist = length
        else:
            stop_dist = distance_to_stop
        t = np.linspace(0, (0 - v0) / acceleration, 120) if acceleration != 0 else np.array([0])
        s = v0 * t + 0.5 * acceleration * t ** 2
        s = np.clip(s, 0, stop_dist)
    else:
        # Block slides down ramp starting from rest
        # v^2 = 2aL; time = v/a
        final_velocity = np.sqrt(2 * acceleration * length) if acceleration > 0 else 0
        time = final_velocity / acceleration if acceleration > 0 else 0
        t = np.linspace(0, time, 120) if time > 0 else np.array([0])
        s = 0.5 * acceleration * t ** 2
        s = np.clip(s, 0, length)

    # Ramp geometry: always top-left (start) to bottom-right (end)
    x0, y0 = 0, length * np.sin(theta_rad)  # top of ramp
    x1, y1 = length * np.cos(theta_rad), 0  # bottom of ramp

    if "up" in direction:
        # Animate from bottom to up (reverse ramp)
        x_block = x1 - s * np.cos(theta_rad)
        y_block = y1 + s * np.sin(theta_rad)
        start_point = (x1, y1)
        end_point = (x0, y0)
        title = "Block Sliding Up an Inclined Plane"
    else:
        # Animate from top to bottom
        x_block = x0 + s * np.cos(theta_rad)
        y_block = y0 - s * np.sin(theta_rad)
        start_point = (x0, y0)
        end_point = (x1, y1)
        title = "Block Sliding Down an Inclined Plane"

    # ---- Animation Section ----
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([x0, x1], [y0, y1], 'k-', lw=4, label="Ramp")
    ax.set_xlim(-0.2 * length, x1 + 0.2 * length)
    ax.set_ylim(-0.2 * length, y0 + 0.3 * length)
    ax.set_xlabel("Horizontal (m)")
    ax.set_ylabel("Vertical (m)")
    ax.set_title(title)
    block, = ax.plot([], [], 'ro', markersize=14, label="Block")

    # Draw ground
    ax.plot([x0, x1, x1 + 0.2 * length], [y1, y1, y1], 'brown', lw=2)

    def init():
        block.set_data([], [])
        return block,

    def animate(i):
        idx = min(i, len(x_block) - 1)
        block.set_data([x_block[idx]], [y_block[idx]])
        return block,

    ani = FuncAnimation(fig, animate, frames=len(x_block), init_func=init, blit=True, interval=25)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer='pillow')
    plt.close(fig)

    st.image(tmpfile.name, caption=title, use_container_width=True)

    # Show FBD and step-by-step if requested by AI
    if use_ai and data.get("free_body_diagram", False):
        with st.expander("Free-Body Diagram (FBD) for this scenario"):
            draw_incline_fbd(
    angle,
    mass,
    mu,
    length,
    show_friction=mu > 0,
    direction=direction,
    initial_velocity=float(data.get("initial_velocity") or 0),
    show_applied=True
)

    # Plots for position/velocity vs time
    with st.expander("View Position & Velocity vs Time Graphs"):
        fig2, ax2 = plt.subplots()
        ax2.plot(t, s, label="Position (m)")
        # For up, velocity is v0 + at; for down, velocity is just at
        if "up" in direction:
            velocity = v0 + acceleration * t
        else:
            velocity = acceleration * t
        ax2.plot(t, velocity, label="Velocity (m/s)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Motion on Inclined Plane")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)
