import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import tempfile

def app(data=None):
    st.title("Animated Free Fall Simulator")

    if data:
        v0 = float(data.get("initial_velocity", 0))
        h0 = float(data.get("height", 0))
        st.markdown("#### Using AI-extracted values:")
        st.markdown(f"- **Initial Height:** `{h0} m`")
        st.markdown(f"- **Initial Velocity:** `{v0} m/s`")
    else:
        h0 = st.slider("Initial Height (m)", 0.0, 100.0, 10.0)
        v0 = st.slider("Initial Velocity (m/s)", -20.0, 20.0, 0.0)  # allow negative (upwards)

    g = 10.0

    discrim = v0**2 + 2 * g * h0
    if discrim < 0:
        st.warning("No valid trajectory. Please check inputs.")
        return

    t_fall = (-v0 + np.sqrt(discrim)) / g if g != 0 else 0
    if t_fall <= 0:
        st.warning("The object does not fall. Check your initial conditions.")
        return

    t = np.linspace(0, t_fall, 120)
    y = h0 + v0 * t - 0.5 * g * t**2

    if np.all(y <= 0) or np.any(np.isnan(y)):
        st.warning("No valid fall to animate. Try changing the parameters.")
        return

    # --- Animation ---
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, max(h0 * 1.1, 1))
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Free Fall (Animated)")

    path, = ax.plot([], [], 'b-', lw=2, label="Path So Far")
    ball, = ax.plot([], [], 'ro', markersize=10, label="Ball")
    arrow = [None]

    def init():
        path.set_data([], [])
        ball.set_data([], [])
        return path, ball

    def animate(i):
        idx = min(i, len(y) - 1)
        # Path so far: vertical
        path.set_data([0] * (idx + 1), y[:idx + 1])
        # Ball position
        ball.set_data([0], [y[idx]])

        # Remove previous arrow
        if arrow[0]:
            try:
                arrow[0].remove()
            except Exception:
                pass

        # Velocity vector (downwards)
        v_curr = v0 - g * t[idx]  # negative = upward
        scale = 0.25 * abs(v_curr)  # scale arrow
        arrow_dir = -1 if v_curr < 0 else 1
        arrow[0] = FancyArrowPatch(
            (0, y[idx]),
            (0, y[idx] + scale * arrow_dir),
            color='green', arrowstyle='->', mutation_scale=18, linewidth=2
        )
        ax.add_patch(arrow[0])
        return path, ball, arrow[0]

    ani = FuncAnimation(fig, animate, frames=len(y), init_func=init, blit=True, interval=20)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer='pillow')
    plt.close(fig)
    st.image(tmpfile.name, caption="Free Fall Animation", use_container_width=True)

    # --- Results ---
    st.markdown("### Results")
    st.markdown(f"- **Time to hit ground:** `{t_fall:.2f}` seconds")
    st.markdown(f"- **Final speed on impact:** `{(v0 - g * t_fall):.2f}` m/s")

    with st.expander("View Calculations and Formulas"):
        st.markdown(r"""
        **Equations Used:**

        - Displacement: $y(t) = h_0 + v_0 t - \frac{1}{2} g t^2$
        - Time of fall: $t = \frac{-v_0 + \sqrt{v_0^2 + 2gh_0}}{g}$
        - Final speed: $v = v_0 - gt$ (negative = downward)
        """)
