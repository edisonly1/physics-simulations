import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile

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
            length = float(data.get("distance", 5) or 5)
        except Exception:
            length = 5

        motion_type = str(data.get("motion_type", "")).lower()
        question_type = str(data.get("question_type", "")).lower()

        # Let user override ramp length if AI did not supply it
        if not data.get("distance"):
            length = st.slider("Ramp Length (m)", 0.1, 20.0, 5.0)

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
    s = np.clip(s, 0, length)  # Clamp block position to ramp length

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

    # Show FBD and step-by-step if requested by AI
    if use_ai and data.get("free_body_diagram", False):
        with st.expander("Free-Body Diagram (FBD) for this scenario"):
            draw_incline_fbd(angle, mass, mu, length, show_friction=mu > 0)
        with st.expander("Step-by-step Acceleration Calculation"):
            st.markdown(f"""
**1. Calculate the force parallel to the ramp:**

$$
F_{{||}} = mg\\sin\\theta = {mass:.2f} \\times {g:.1f} \\times \\sin({angle:.1f}^\\circ) = {f_parallel:.2f}\\ \\text{{N}}
$$

**2. Calculate the normal force:**

$$
F_N = mg\\cos\\theta = {mass:.2f} \\times {g:.1f} \\times \\cos({angle:.1f}^\\circ) = {f_normal:.2f}\\ \\text{{N}}
$$

**3. Calculate the frictional force:**

$$
f_k = \\mu F_N = {mu:.2f} \\times {f_normal:.2f} = {f_friction:.2f}\\ \\text{{N}}
$$

**4. Calculate the net force down the ramp:**

$$
F_{{net}} = F_{{||}} - f_k = {f_parallel:.2f} - {f_friction:.2f} = {f_net:.2f}\\ \\text{{N}}
$$

**5. Finally, calculate the acceleration:**

$$
a = \\frac{{F_{{net}}}}{{m}} = \\frac{{{f_net:.2f}}}{{{mass:.2f}}} = {acceleration:.2f}\\ \\text{{m/s}}^2
$$
""")

    # Plots for position/velocity vs time
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


def draw_incline_fbd(angle_deg=30, mass=2, mu=0, length=5, show_friction=False):
    g = 10.0
    theta = np.radians(angle_deg)
    # Ramp: top-left to bottom-right
    x0, y0 = 0, length * np.sin(theta)
    x1, y1 = length, 0

    # Block position (1/3 along ramp)
    frac = 1/3
    xb = x0 + frac * (x1 - x0)
    yb = y0 + frac * (y1 - y0)
    block_size = 0.3

    # Block (rotated square)
    block_angle = -theta
    block = np.array([
        [-block_size/2, -block_size/2],
        [ block_size/2, -block_size/2],
        [ block_size/2,  block_size/2],
        [-block_size/2,  block_size/2],
        [-block_size/2, -block_size/2]
    ])
    rot = np.array([
        [np.cos(block_angle), -np.sin(block_angle)],
        [np.sin(block_angle),  np.cos(block_angle)]
    ])
    block_rot = block @ rot.T + [xb, yb]

    # Forces
    mg = mass * g
    fn = mg * np.cos(theta)
    fp = mg * np.sin(theta)
    f_friction = mu * fn if show_friction and mu > 0 else 0

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([x0, x1], [y0, y1], color="k", lw=4)
    ax.plot([x1 - 1, x1 + 1], [y1, y1], color="brown", lw=2)
    ax.plot(block_rot[:, 0], block_rot[:, 1], color="royalblue", lw=2)
    ax.fill(block_rot[:, 0], block_rot[:, 1], "royalblue", alpha=0.7)

    block_center = np.array([xb, yb])
    # Gravity (down)
    ax.arrow(*block_center, 0, -mg*0.15, head_width=0.13, head_length=0.18, fc='green', ec='green', lw=3, length_includes_head=True)
    ax.text(block_center[0], block_center[1] - mg*0.18 - 0.2, r"$mg$", color="green", fontsize=18)
    # Normal (perpendicular)
    nx = np.sin(theta)
    ny = np.cos(theta)
    ax.arrow(*block_center, nx*fn*0.06, ny*fn*0.06, head_width=0.13, head_length=0.18, fc='orange', ec='orange', lw=3, length_includes_head=True)
    ax.text(block_center[0] + nx*fn*0.09 + 0.12, block_center[1] + ny*fn*0.09, r"$N$", color="orange", fontsize=18)
    # Parallel (down ramp)
    px = np.cos(theta)
    py = -np.sin(theta)
    ax.arrow(*block_center, px*fp*0.08, py*fp*0.08, head_width=0.13, head_length=0.18, fc='red', ec='red', lw=3, length_includes_head=True)
    ax.text(block_center[0] + px*fp*0.11, block_center[1] + py*fp*0.11 - 0.1, r"$F_{\parallel}$", color="red", fontsize=18)
    # Friction (if present, up the ramp)
    if show_friction and mu > 0:
        ax.arrow(*block_center, -px*f_friction*0.06, -py*f_friction*0.06, head_width=0.13, head_length=0.18, fc='brown', ec='brown', lw=3, length_includes_head=True)
        ax.text(block_center[0] - px*f_friction*0.09, block_center[1] - py*f_friction*0.09 + 0.1, r"$f_k$", color="brown", fontsize=18)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-length*0.2, length*1.2)
    ax.set_ylim(-length*0.5, y0 + block_size*2)
    ax.set_title("Free-Body Diagram (FBD) for Block on Incline", fontsize=18, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)
