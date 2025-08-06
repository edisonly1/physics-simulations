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

    # --- Input Handling ---
    if use_ai:
        st.markdown("#### Using AI-extracted values:")
        try:
            angle = float(data.get("angle", 30))
        except Exception:
            angle = 30

        try:
            mass = float(data.get("mass", 1))
        except Exception:
            mass = 1

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
            mu = float(fc) if fc is not None else 0.2

        try:
            length = float(data.get("length", 5) or 5)
        except Exception:
            length = 5

        direction_raw = str(data.get("direction", "down")).lower()
        if "up" in direction_raw:
            direction = "up"
        else:
            direction = "down"

        v0 = float(data.get("initial_velocity") or 0)
        vf = float(data.get("final_velocity") or 0)
        show_fbd = data.get("free_body_diagram", False)
        show_applied = abs(v0) > 0
    else:
        st.markdown("#### Interactive mode: Use the sliders below to create your own ramp scenario.")
        mass = st.slider("Mass (kg)", 0.1, 10.0, 1.0)
        angle = st.slider("Incline Angle (°)", 0.0, 90.0, 30.0)
        mu = st.slider("Friction Coefficient (μ)", 0.0, 1.0, 0.0)
        length = st.slider("Ramp Length (m)", 0.1, 20.0, 5.0)
        direction = st.selectbox("Direction", ["down", "up"]).lower()
        v0 = st.slider("Initial velocity (m/s)", 0.0, 20.0, 0.0)
        vf = 0.0
        show_fbd = st.checkbox("Show Free-Body Diagram", value=True)
        show_applied = st.checkbox("Show applied (push) force?", value=(abs(v0) > 0))

    theta_rad = np.radians(angle)
    f_parallel = mass * g * np.sin(theta_rad)
    f_normal = mass * g * np.cos(theta_rad)
    f_friction = mu * f_normal

    # --- Constant velocity logic ---
    is_constant_velocity = abs(v0 - vf) < 1e-6 and v0 > 0

    if is_constant_velocity:
        f_applied = f_parallel + f_friction if "up" in direction else f_parallel - f_friction
        st.markdown("### Results (Constant Velocity)")
        st.markdown("**The block moves at a constant speed.** Net force is zero; applied force exactly balances gravity and friction.")
        st.markdown(f"- **Force required to move at constant velocity:** `{f_applied:.2f} N`")
        st.markdown(f"- **Friction force:** `{f_friction:.2f} N`")
        st.markdown(f"- **Gravity component down ramp:** `{f_parallel:.2f} N`")
        st.info("No animation required as speed is constant.")

        # FBD (mode-specific)
        if show_fbd:
            with st.expander("Free-Body Diagram (FBD) for this scenario"):
                draw_incline_fbd(
                    angle,
                    mass,
                    mu,
                    length,
                    show_friction=mu > 0,
                    direction=direction,
                    initial_velocity=v0,
                    show_applied=show_applied
                )
        return

    # --- Standard kinematics/animation ---
    if "up" in direction:
        f_net = -f_parallel - f_friction
    else:
        f_net = f_parallel - f_friction
    acceleration = f_net / mass

    if "up" in direction:
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
        final_velocity = np.sqrt(2 * acceleration * length) if acceleration > 0 else 0
        time = final_velocity / acceleration if acceleration > 0 else 0
        t = np.linspace(0, time, 120) if time > 0 else np.array([0])
        s = v0 * t + 0.5 * acceleration * t ** 2
        s = np.clip(s, 0, length)

    # --- Geometry for animation ---
    x0, y0 = 0, length * np.sin(theta_rad)
    x1, y1 = length * np.cos(theta_rad), 0

    if "up" in direction:
        x_block = x1 - s * np.cos(theta_rad)
        y_block = y1 + s * np.sin(theta_rad)
        title = "Block Sliding Up an Inclined Plane"
    else:
        x_block = x0 + s * np.cos(theta_rad)
        y_block = y0 - s * np.sin(theta_rad)
        title = "Block Sliding Down an Inclined Plane"

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([x0, x1], [y0, y1], 'k-', lw=4, label="Ramp")
    ax.set_xlim(-0.2 * length, x1 + 0.2 * length)
    ax.set_ylim(-0.2 * length, y0 + 0.3 * length)
    ax.set_xlabel("Horizontal (m)")
    ax.set_ylabel("Vertical (m)")
    ax.set_title(title)
    block, = ax.plot([], [], 'ro', markersize=14, label="Block")
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

    # --- FBD: manual and AI mode supported ---
    if show_fbd:
        with st.expander("Free-Body Diagram (FBD) for this scenario"):
            draw_incline_fbd(
                angle,
                mass,
                mu,
                length,
                show_friction=mu > 0,
                direction=direction,
                initial_velocity=v0,
                show_applied=show_applied
            )

    # --- Plots for position/velocity vs time ---
    with st.expander("View Position & Velocity vs Time Graphs"):
        fig2, ax2 = plt.subplots()
        ax2.plot(t, s, label="Position (m)")
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
