# FrictionModels.py
# AP Physics 1 – Static & Kinetic Friction (with optional incline)
# Streamlit module: shows threshold from no motion (static) to motion (kinetic)
# Animation uses a fixed rendering FPS so the visual speed does not depend on dt.

from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import streamlit as st


def app():
    st.title("Friction Models: Static → Kinetic")
    st.caption("Purpose: Show the threshold between **no motion** and **motion** under friction. "
               "Visuals ramp the applied force until motion begins, then continue under kinetic friction.")

    # -----------------------
    # Sidebar: Inputs
    # -----------------------
    st.sidebar.header("Inputs")
    col_mu = st.sidebar.columns(2)
    mu_s = col_mu[0].number_input("μs (static)", min_value=0.0, value=0.40, step=0.01, format="%.3f")
    mu_k = col_mu[1].number_input("μk (kinetic)", min_value=0.0, value=0.30, step=0.01, format="%.3f")

    if mu_k > mu_s:
        st.sidebar.warning("Tip: Usually μk ≤ μs. Using μk > μs may look odd physically.")

    mass = st.sidebar.number_input("Mass m (kg)", min_value=0.1, value=10.0, step=0.1, format="%.2f")
    g = st.sidebar.number_input("g (m/s²)", min_value=0.1, value=9.8, step=0.1, format="%.2f")

    st.sidebar.markdown("---")
    use_incline = st.sidebar.checkbox("Combine with Inclined Plane", value=False)
    if use_incline:
        angle_deg = st.sidebar.slider("Incline angle θ (deg)", 0.0, 60.0, 20.0, 0.5)
        direction = st.sidebar.selectbox("Applied force direction", ["Up-slope", "Down-slope"])
    else:
        angle_deg = 0.0
        direction = "Right →"

    st.sidebar.markdown("---")
    # Force ramp parameters
    F_max = st.sidebar.number_input("Max applied force Fₐ, max (N)", min_value=0.0, value=300.0, step=10.0)
    ramp_time = st.sidebar.number_input("Ramp time to Fₐ, max (s)", min_value=0.1, value=5.0, step=0.1)
    hold_time = st.sidebar.number_input("Hold at Fₐ, max (s)", min_value=0.0, value=2.0, step=0.1)

    st.sidebar.markdown("---")
    # Animation controls (visual smoothness independent of physics dt)
    fps = st.sidebar.slider("Animation FPS", 10, 60, 40, step=5)
    sim_dt = st.sidebar.number_input("Simulation Δt (s)", min_value=0.001, value=0.002, step=0.001, format="%.3f")
    st.sidebar.caption("The animation always renders at the chosen FPS; changing Δt only affects numerical accuracy.")

    # Classroom example button
    if st.sidebar.button("Load Classroom Example (m=10 kg, μs=0.4, μk=0.3)"):
        st.experimental_set_query_params(example="1")
        st.rerun()

    # -----------------------
    # Pre-compute constants
    # -----------------------
    theta = np.deg2rad(angle_deg)
    if use_incline:
        N = mass * g * np.cos(theta)
        # Coordinate axis is chosen along "applied direction"
        dir_sign = +1 if direction.startswith("Up") else -1  # +: up-slope,  -: down-slope
        F_g = -dir_sign * mass * g * np.sin(theta)           # gravity component along the axis
        axis_label = "along slope (+ " + ("up" if dir_sign > 0 else "down") + ")"
    else:
        N = mass * g
        dir_sign = +1
        F_g = 0.0
        axis_label = "horizontal (+ right)"

    F_static_max = mu_s * N
    F_kinetic_mag = mu_k * N

    # Threshold estimate (when object *begins* to move from rest)
    # From rest, friction can match until |F_app + F_g| > μs N.
    # Solve |F_app_start + F_g| = μs N → F_app_start = μs N - F_g with sign chosen to increase |·|.
    # We present the smallest nonnegative F_app that triggers motion in the chosen + axis.
    if (F_g >= 0):  # gravity helps the + direction
        F_start = max(0.0, F_static_max - F_g)
    else:           # gravity opposes the + direction
        F_start = F_static_max + (-F_g)

    with st.expander("Formulas and thresholds", expanded=True):
        st.latex(r"F_N = mg\cos\theta \quad\text{(horizontal: }\theta=0\Rightarrow F_N=mg\text{)}")
        st.latex(r"F_{s,\max} = \mu_s F_N \qquad F_k = \mu_k F_N")
        st.latex(r"\text{From rest: object moves when }|F_{\rm app} + F_g| > \mu_s F_N")
        st.write(f"**Normal force** N = {N:,.2f} N")
        st.write(f"**Static limit** μs·N = {F_static_max:,.2f} N • **Kinetic** μk·N = {F_kinetic_mag:,.2f} N")
        if use_incline:
            sgn = "helps" if F_g > 0 else ("opposes" if F_g < 0 else "neither helps nor opposes")
            st.write(f"Gravity component along the axis: **Fg = {F_g:,.2f} N** ({sgn}).")
        st.info(f"Estimated **start of motion** at applied force ≈ **{F_start:,.2f} N** "
                f"(in the chosen + direction, {axis_label}).")

    # -----------------------
    # Build the time base
    # -----------------------
    T_total = ramp_time + hold_time
    steps = int(np.ceil(T_total / sim_dt)) + 1
    t = np.linspace(0, T_total, steps)

    # Applied force vs time: linear ramp to F_max, then hold
    F_app = np.minimum(F_max, F_max * (t / max(ramp_time, 1e-9)))

    # -----------------------
    # Simulate 1D translational motion with Coulomb friction
    # -----------------------
    x = np.zeros_like(t)    # position (m) – purely visual here
    v = np.zeros_like(t)    # velocity (m/s)
    a = np.zeros_like(t)    # acceleration (m/s^2)
    F_fric = np.zeros_like(t)
    F_net = np.zeros_like(t)
    regime = np.zeros_like(t, dtype=int)  # 0=static, 1=kinetic

    v_eps = 1e-6
    moved_once = False
    t_transition = None

    for i in range(1, steps):
        # Forces excluding friction along the axis
        F_nf = F_app[i] + F_g

        # Decide static vs kinetic
        if abs(v[i-1]) < v_eps:  # treat as "at rest"
            if abs(F_nf) <= F_static_max:
                # Static friction balances all
                F_f = -F_nf
                a_i = 0.0
                regime[i] = 0
            else:
                # Static breaks → kinetic friction opposes impending motion (sign of F_nf)
                sign_impending = np.sign(F_nf) if F_nf != 0 else +1.0
                F_f = -sign_impending * F_kinetic_mag
                a_i = (F_nf + F_f) / mass
                regime[i] = 1
                if not moved_once:
                    moved_once = True
                    t_transition = t[i]
        else:
            # Already moving → kinetic friction opposes velocity
            sign_v = np.sign(v[i-1])
            F_f = -sign_v * F_kinetic_mag if sign_v != 0 else 0.0
            a_i = (F_nf + F_f) / mass
            regime[i] = 1

        # Integrate
        v[i] = v[i-1] + a_i * sim_dt
        x[i] = x[i-1] + v[i] * sim_dt
        a[i] = a_i
        F_fric[i] = F_f
        F_net[i] = F_nf + F_f

    # -----------------------
    # Charts (forces & velocity)
    # -----------------------
    c1, c2 = st.columns([1.2, 1.0])

    with c1:
        figF, axF = plt.subplots(figsize=(7.2, 3.4))
        axF.plot(t, F_app, label="Applied force $F_{app}$")
        axF.plot(t, F_fric, label="Friction $F_{fr}$")
        axF.plot(t, F_net, label="Net force $F_{net}$", linestyle="--")
        axF.axhline(+F_static_max, linewidth=1, alpha=0.5)
        axF.axhline(-F_static_max, linewidth=1, alpha=0.5)
        axF.text(0.01, F_static_max*1.02, "± μsN", fontsize=9, va="bottom")
        if t_transition is not None:
            axF.axvline(t_transition, color="k", linewidth=1)
            axF.text(t_transition, 0.02, " static→kinetic ", rotation=90,
                     transform=axF.get_xaxis_transform(), fontsize=9, va="bottom", ha="center")
        axF.set_title("Forces vs Time")
        axF.set_xlabel("time (s)")
        axF.set_ylabel("Force (N)")
        axF.grid(True, alpha=0.25)
        axF.legend(loc="upper left")
        st.pyplot(figF, clear_figure=True)
        plt.close(figF)

    with c2:
        figV, axV = plt.subplots(figsize=(6.0, 3.4))
        axV.plot(t, v, label="velocity $v$")
        axV.set_title("Velocity vs Time")
        axV.set_xlabel("time (s)")
        axV.set_ylabel("v (m/s)")
        if t_transition is not None:
            axV.axvline(t_transition, linewidth=1)
        axV.grid(True, alpha=0.25)
        axV.legend(loc="upper left")
        st.pyplot(figV, clear_figure=True)
        plt.close(figV)

    # -----------------------
    # Animation: block + force arrows
    # -----------------------
    st.subheader("Animation")
    st.caption("Applied force ramps up gradually. Watch static friction match it until the threshold, then motion under kinetic friction.")

    start = st.button("▶ Start / Replay")
    # Visual layout constants
    track_len = 6.0            # meters (for mapping x → pixels)
    px_per_m = 70.0
    block_w, block_h = 0.8, 0.5

    # Set up the figure once
    figA, axA = plt.subplots(figsize=(9.0, 3.5))
    axA.set_xlim(0, track_len)
    axA.set_ylim(-1.2, 1.8)
    axA.set_aspect('auto')
    axA.axis('off')

    # Ground / slope
    if use_incline:
        # draw a sloped line
        x0, y0 = 0.2, 0.3
        x1 = track_len - 0.2
        slope = np.tan(theta)
        xs = np.linspace(x0, x1, 200)
        ys = y0 + slope * (xs - x0)
        axA.plot(xs, ys, linewidth=3)
        # helper to map 1D position along axis to (x,y) on slope
        def pos_to_xy(s):
            # s measured along axis; place block center near left
            x = x0 + (s + 0.5) * np.cos(theta)
            y = y0 + (s + 0.5) * np.sin(theta)
            return x, y
        axis_dir_text = "(+ up-slope)" if dir_sign > 0 else "(+ down-slope)"
    else:
        axA.plot([0.1, track_len-0.1], [0.3, 0.3], linewidth=3)
        def pos_to_xy(s):
            return 0.7 + s, 0.55
        axis_dir_text = "(+ right)"

    axA.text(0.05, 1.55, f"Axis {axis_dir_text}", fontsize=10)

    # Block patch (we place by updating its transform each frame)
    bx, by = pos_to_xy(0.0)
    block = Rectangle((bx - block_w/2, by - block_h/2), block_w, block_h, linewidth=1.5, edgecolor='k', facecolor='none')
    axA.add_patch(block)

    arr_app = make_arrow(axA)
    arr_fric = make_arrow(axA)
    arr_N    = make_arrow(axA)
    arr_W    = make_arrow(axA)


    status_text = axA.text(track_len*0.55, 1.30, "", fontsize=12, fontweight='bold')

    # helper to draw/update an arrow from (x,y) along unit (ux,uy) with magnitude scaled visually


    def make_arrow(ax):
        a = FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=18, linewidth=2)
        ax.add_patch(a)
        return a

    def update_arrow(arrow, x, y, ux, uy, magnitude, scale=0.004):
        # Draw arrow from (x,y) to (x+L*ux, y+L*uy)
        L = abs(magnitude) * scale
        if magnitude < 0:
            ux, uy = -ux, -uy
        arrow.set_positions((x, y), (x + L*ux, y + L*uy))


    # One static render (initial)
    st.pyplot(figA)  # first draw (blank state)
    plt.close(figA)  # Streamlit copies the figure; we'll recreate each frame below

    if start:
        placeholder = st.empty()
        # Recreate a figure for the animation loop (avoids Streamlit reuse warnings)
        figA, axA = plt.subplots(figsize=(9.0, 3.5))
        axA.set_xlim(0, track_len)
        axA.set_ylim(-1.2, 1.8)
        axA.axis('off')

        # Ground / slope again
        if use_incline:
            x0, y0 = 0.2, 0.3
            x1 = track_len - 0.2
            slope = np.tan(theta)
            xs = np.linspace(x0, x1, 200)
            ys = y0 + slope * (xs - x0)
            axA.plot(xs, ys, linewidth=3)
            def pos_to_xy(s):
                x = x0 + (s + 0.5) * np.cos(theta)
                y = y0 + (s + 0.5) * np.sin(theta)
                return x, y
            axis_dir_text = "(+ up-slope)" if dir_sign > 0 else "(+ down-slope)"
        else:
            axA.plot([0.1, track_len-0.1], [0.3, 0.3], linewidth=3)
            def pos_to_xy(s):
                return 0.7 + s, 0.55
            axis_dir_text = "(+ right)"
        axA.text(0.05, 1.55, f"Axis {axis_dir_text}", fontsize=10)

        block = Rectangle((0, 0), block_w, block_h, linewidth=1.5, edgecolor='k', facecolor='none')
        axA.add_patch(block)

        arr_app = make_arrow(axA)
        arr_fric = make_arrow(axA)
        arr_N    = make_arrow(axA)
        arr_W    = make_arrow(axA)

        status_text = axA.text(track_len*0.55, 1.30, "", fontsize=12, fontweight='bold')

        # Render frames at fixed FPS by skipping/merging sim steps
        frame_dt = 1.0 / fps
        next_frame_t = 0.0
        i = 0
        while i < len(t):
            # Find index closest to next_frame_t
            while i < len(t)-1 and t[i] < next_frame_t:
                i += 1

            # Map 1D position x→(X,Y)
            s_pos = np.clip(x[i], 0.0, track_len - 1.2)
            bx, by = pos_to_xy(s_pos)
            block.set_xy((bx - block_w/2, by - block_h/2))

            # Build local tangent & normal unit vectors for arrows
            if use_incline:
                ux, uy = np.cos(theta)*dir_sign, np.sin(theta)*dir_sign           # +axis direction
                nx, ny = -np.sin(theta), np.cos(theta)                            # outward normal
            else:
                ux, uy = 1.0, 0.0
                nx, ny = 0.0, 1.0

            # Arrow anchor (block center)
            cx, cy = bx, by

            # Update arrows
            update_arrow(arr_app, cx, cy, ux, uy, F_app[i], scale=0.004)
            update_arrow(arr_fric, cx, cy, ux, uy, F_fric[i], scale=0.004)
            update_arrow(arr_N,   cx, cy, nx, ny, N,         scale=0.004)
            update_arrow(arr_W,   cx, cy, 0.0, -1.0, mass*g, scale=0.004)

            # Status label
            status_text.set_text(f"Regime: {'Static' if regime[i]==0 else 'Kinetic'}   "
                                 f"F_app={F_app[i]:.0f} N,  F_fr={F_fric[i]:.0f} N")

            axA.set_title("Block + Force Vectors")
            placeholder.pyplot(figA)
            next_frame_t += frame_dt
            time.sleep(max(0.0, frame_dt * 0.85))  # small sleep to avoid pegging CPU

        plt.close(figA)

    # -----------------------
    # Classroom prompt
    # -----------------------
    st.markdown("""
**Try this classroom example:**  
*“How much force is needed to start moving a 10 kg crate with μs = 0.4, μk = 0.3?”*  
Use the inputs on the left. The vertical dashed line on the force plot marks the
static→kinetic transition.
""")


# If you want to run this file directly (useful for quick testing):
if __name__ == "__main__":
    import streamlit.web.bootstrap as boot
    boot.run("FrictionModels.py", "", [], {})
