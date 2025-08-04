import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import tempfile

def app(data=None):
    st.title("Animated Projectile Motion Simulator")

    # --- Input Handling ---
    v0 = float(data.get("initial_velocity" or 20)) if data else st.slider("Initial Velocity (m/s)", 0.0, 50.0, 25.0, step=0.1)
    angle_deg = float(data.get("angle" or 45)) if data else st.slider("Launch Angle (degrees)", 0.0, 90.0, 45.0, step=0.1)
    h0 = float(data.get("height" or 0)) if data else st.slider("Initial Height (m)", 0.0, 10.0, 0.0, step=0.1)
    question_type = data.get("question_type", "").lower() if data else ""

    if data:
        st.markdown(f"**Initial Velocity:** {v0} m/s  \n**Angle:** {angle_deg}°  \n**Height:** {h0} m")
        if question_type in ["range", "kinematics"]:
            # Use variable names for clarity
            theta = np.radians(angle_deg)
            v0x = v0 * np.cos(theta)
            v0y = v0 * np.sin(theta)
            g = 10.0

            sqrt_term = np.sqrt(v0y ** 2 + 2 * g * h0)
            t_range = (v0y + sqrt_term) / g
            range_val = v0x * t_range

            st.success(
                f"""
                **Step-by-step Range Calculation:**

                Range = $v_{{0x}} \\times t_{{\\text{{flight}}}}$

                $v_{{0x}} = v_0 \\cos\\theta = {v0:.2f} \\times \\cos({angle_deg:.2f}^\\circ) = {v0x:.2f}$ m/s  
                $v_{{0y}} = v_0 \\sin\\theta = {v0:.2f} \\times \\sin({angle_deg:.2f}^\\circ) = {v0y:.2f}$ m/s  

                $t_{{\\text{{flight}}}} = \\frac{{v_{{0y}} + \\sqrt{{v_{{0y}}^2 + 2gh_0}}}}{{g}} = \\frac{{{v0y:.2f} + \\sqrt{{({v0y:.2f})^2 + 2 \\times {g:.1f} \\times {h0:.2f}}}}}{{{g:.1f}}} = {t_range:.2f}$ s

                **Final range:**  
                ${v0x:.2f} \\times {t_range:.2f} = {range_val:.2f}$ meters
                """
            )
    else:
        st.info("Adjust the sliders to see the projectile's flight.")

    # --- Physics Calculation ---
    g = 10.0
    theta = np.radians(angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    discrim = vy**2 + 2 * g * h0
    t_flight = (vy + np.sqrt(discrim)) / g if discrim >= 0 else 0

    if t_flight <= 0 or (v0 == 0 and h0 == 0):
        st.warning("The object does not move. Please check your initial conditions.")
        return

    t_vals = np.linspace(0, t_flight, 180)
    x_vals = vx * t_vals
    y_vals = h0 + vy * t_vals - 0.5 * g * t_vals**2

    n_frames = min(len(x_vals), len(y_vals))
    if n_frames < 2 or np.all(y_vals <= 0) or np.any(np.isnan(x_vals)) or np.any(np.isnan(y_vals)):
        st.warning("No valid trajectory to animate. Try changing the parameters.")
        return

    fig, ax = plt.subplots()
    ax.set_xlim(0, max(np.max(x_vals) * 1.05, 1))
    ax.set_ylim(0, max(np.max(y_vals) * 1.05, 1))
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Projectile Path (Animated)")

    # Static background trajectory
    ax.plot(x_vals, y_vals, '--', color='lightgray', label="Full Trajectory")

    line, = ax.plot([], [], 'b-', lw=2, label="Path So Far")
    point, = ax.plot([], [], 'ro', markersize=8, label="Ball")
    arrow = [None]  # To update velocity vector

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def animate(i):
        idx = min(i, n_frames - 1)
        t = t_vals[idx]

        # Path trail up to current frame
        line.set_data(x_vals[:idx + 1], y_vals[:idx + 1])
        # Current position of the ball
        point.set_data(x_vals[idx:idx + 1], y_vals[idx:idx + 1])

        # Remove the previous arrow, if any
        if arrow[0] is not None:
            try:
                arrow[0].remove()
            except Exception:
                pass

        # Instantaneous velocity
        vx_i = vx
        vy_i = vy - g * t
        scale = 0.2 * np.sqrt(vx_i**2 + vy_i**2)  # Scaling for visual clarity

        # Add velocity arrow
        arrow[0] = FancyArrowPatch(
            (x_vals[idx], y_vals[idx]),
            (x_vals[idx] + scale * vx_i / (np.sqrt(vx_i**2 + vy_i**2) + 1e-6), 
             y_vals[idx] + scale * vy_i / (np.sqrt(vx_i**2 + vy_i**2) + 1e-6)),
            color='green', arrowstyle='->', mutation_scale=20, linewidth=2
        )
        ax.add_patch(arrow[0])
        return line, point, arrow[0]

    ani = FuncAnimation(fig, animate, frames=n_frames, init_func=init, blit=True, interval=20)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer='pillow')
    plt.close(fig)

    st.image(tmpfile.name, caption="Projectile Animation", use_container_width=True)

    # --- Results ---
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
