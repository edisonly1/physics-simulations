# FrictionModels.py
# AP Physics 1 – Static & Kinetic Friction (with optional incline)
# Streamlit module: shows threshold from no motion (static) to motion (kinetic)

from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import streamlit as st

def app():
    st.title("Friction Models: Static → Kinetic")
    st.caption("Ramp the applied force and watch the transition from static friction to kinetic friction. "
               "Rendering runs at fixed FPS, so visuals are independent of Δt.")

    # ---------------- Sidebar ----------------
    st.sidebar.header("Inputs")
    col_mu = st.sidebar.columns(2)
    mu_s = col_mu[0].number_input("μs (static)", min_value=0.0, value=0.40, step=0.01, format="%.3f")
    mu_k = col_mu[1].number_input("μk (kinetic)", min_value=0.0, value=0.30, step=0.01, format="%.3f")
    if mu_k > mu_s:
        st.sidebar.warning("Usually μk ≤ μs. Using μk > μs may look unphysical.")

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
    F_max = st.sidebar.number_input("Max applied force Fₐ,max (N)", min_value=0.0, value=300.0, step=10.0)
    ramp_time = st.sidebar.number_input("Ramp time to Fₐ,max (s)", min_value=0.1, value=5.0, step=0.1)
    hold_time = st.sidebar.number_input("Hold at Fₐ,max (s)", min_value=0.0, value=2.0, step=0.1)

    st.sidebar.markdown("---")
    fps = st.sidebar.slider("Animation FPS", 10, 60, 40, step=5)
    sim_dt = st.sidebar.number_input("Simulation Δt (s)", min_value=0.001, value=0.002, step=0.001, format="%.3f")
    st.sidebar.caption("Animation renders at fixed FPS; Δt only affects the physics integration accuracy.")

    if st.sidebar.button("Load Classroom Example (m=10 kg, μs=0.4, μk=0.3)"):
        st.experimental_set_query_params(example="1")
        st.rerun()

    # -------------- Physics setup --------------
    theta = np.deg2rad(angle_deg)
    if use_incline:
        N = mass * g * np.cos(theta)
        dir_sign = +1 if direction.startswith("Up") else -1
        F_g = -dir_sign * mass * g * np.sin(theta)   # along the chosen + axis
        axis_label = "along slope (+ " + ("up" if dir_sign > 0 else "down") + ")"
    else:
        N = mass * g
        dir_sign = +1
        F_g = 0.0
        axis_label = "horizontal (+ right)"

    F_static_max = mu_s * N
    F_kinetic_mag = mu_k * N

    # Estimated start-of-motion threshold in + axis
    if (F_g >= 0):
        F_start = max(0.0, F_static_max - F_g)
    else:
        F_start = F_static_max + (-F_g)

    with st.expander("Formulas & thresholds", expanded=True):
        st.latex(r"F_N = mg\cos\theta,\quad F_{s,\max}=\mu_s F_N,\quad F_k=\mu_k F_N")
        st.latex(r"\text{From rest: moves when }|F_{\rm app}+F_g|>\mu_s F_N")
        st.write(f"Normal force N = {N:,.2f} N")
        st.write(f"Static limit μs·N = {F_static_max:,.2f} N • Kinetic μk·N = {F_kinetic_mag:,.2f} N")
        if use_incline:
            sgn = "helps +" if F_g > 0 else ("opposes +" if F_g < 0 else "neutral")
            st.write(f"Gravity component along axis: Fg = {F_g:,.2f} N ({sgn}).")
        st.info(f"Estimated start of motion at F_app ≈ **{F_start:,.2f} N** ({axis_label}).")

    # -------------- Time base & applied force --------------
    T_total = ramp_time + hold_time
    steps = int(np.ceil(T_total / sim_dt)) + 1
    t = np.linspace(0, T_total, steps)
    F_app = np.minimum(F_max, F_max * (t / max(ramp_time, 1e-9)))

    # -------------- Simulation (1D) --------------
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    F_fric = np.zeros_like(t)
    F_net = np.zeros_like(t)
    regime = np.zeros_like(t, dtype=int)  # 0 static, 1 kinetic

    v_eps = 1e-6
    moved_once = False
    t_transition = None

    for i in range(1, steps):
        F_nf = F_app[i] + F_g  # non-friction forces along axis

        if abs(v[i-1]) < v_eps:  # treat as at rest
            if abs(F_nf) <= F_static_max:
                F_f = -F_nf
                a_i = 0.0
                regime[i] = 0
            else:
                sign_impending = np.sign(F_nf) if F_nf != 0 else +1.0
                F_f = -sign_impending * F_kinetic_mag
                a_i = (F_nf + F_f) / mass
                regime[i] = 1
                if not moved_once:
                    moved_once = True
                    t_transition = t[i]
        else:
            sign_v = np.sign(v[i-1])
            F_f = -sign_v * F_kinetic_mag if sign_v != 0 else 0.0
            a_i = (F_nf + F_f) / mass
            regime[i] = 1

        v[i] = v[i-1] + a_i * sim_dt
        x[i] = x[i-1] + v[i] * sim_dt
        a[i] = a_i
        F_fric[i] = F_f
        F_net[i] = F_nf + F_f

    # -------------- Plots --------------
    c1, c2 = st.columns([1.2, 1.0])

    with c1:
        figF, axF = plt.subplots(figsize=(7.2, 3.4))
        axF.plot(t, F_app, label="Applied $F_{app}$")
        axF.plot(t, F_fric, label="Friction $F_{fr}$")
        axF.plot(t, F_net, label="Net $F_{net}$", linestyle="--")
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
        if t_transition is not None:
            axV.axvline(t_transition, linewidth=1)
        axV.set_title("Velocity vs Time")
        axV.set_xlabel("time (s)")
        axV.set_ylabel("v (m/s)")
        axV.grid(True, alpha=0.25)
        axV.legend(loc="upper left")
        st.pyplot(figV, clear_figure=True)
        plt.close(figV)

    # -------------- Animation helpers --------------
    def make_arrow(ax):
        a = FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=18, linewidth=2)
        ax.add_patch(a)
        return a

    def update_arrow(arrow, x0, y0, ux, uy, magnitude, scale=0.004):
        L = abs(magnitude) * scale
        if magnitude < 0:
            ux, uy = -ux, -uy
        arrow.set_positions((x0, y0), (x0 + L*ux, y0 + L*uy))

    # -------------- Animation (static preview then live) --------------
    st.subheader("Animation")
    st.caption("Force vectors shown: applied, friction, normal, weight.")

    track_len = 6.0
    block_w, block_h = 0.8, 0.5
    track_len = 6.0
    block_w, block_h = 0.8, 0.5

    # Ledge limits for the 1-D axis used by x[]
    s_min = 0.0
    s_max = track_len - 1.2   # same clearance you used before to keep the block on-screen


    # Static preview figure
    figA, axA = plt.subplots(figsize=(9.0, 3.5))
    axA.set_xlim(0, track_len)
    axA.set_ylim(-1.2, 1.8)
    axA.axis('off')

    if use_incline:
        x0, y0 = 0.2, 0.3
        x1 = track_len - 0.2
        slope = np.tan(theta)
        xs = np.linspace(x0, x1, 200)
        ys = y0 + slope * (xs - x0)
        axA.plot(xs, ys, linewidth=3)
        def pos_to_xy(s):
            return x0 + (s + 0.5) * np.cos(theta), y0 + (s + 0.5) * np.sin(theta)
        axis_dir_text = "(+ up-slope)" if dir_sign > 0 else "(+ down-slope)"
        bx, by = pos_to_xy(0.0)
    else:
        axA.plot([0.1, track_len-0.1], [0.3, 0.3], linewidth=3)
        def pos_to_xy(s):
            return 0.7 + s, 0.55
        axis_dir_text = "(+ right)"
        bx, by = pos_to_xy(0.0)

    axA.text(0.05, 1.55, f"Axis {axis_dir_text}", fontsize=10)
    block = Rectangle((bx - block_w/2, by - block_h/2), block_w, block_h, linewidth=1.5, edgecolor='k', facecolor='none')
    axA.add_patch(block)
    st.pyplot(figA)
    plt.close(figA)

    start = st.button("▶ Start / Replay")
    if not start:
        return

    # Live animation
    placeholder = st.empty()
    figA, axA = plt.subplots(figsize=(9.0, 3.5))
    axA.set_xlim(0, track_len)
    axA.set_ylim(-1.2, 1.8)
    axA.axis('off')

    if use_incline:
        x0, y0 = 0.2, 0.3
        x1 = track_len - 0.2
        slope = np.tan(theta)
        xs = np.linspace(x0, x1, 200)
        ys = y0 + slope * (xs - x0)
        axA.plot(xs, ys, linewidth=3)
        def pos_to_xy(s):
            return x0 + (s + 0.5) * np.cos(theta), y0 + (s + 0.5) * np.sin(theta)
        axis_dir_text = "(+ up-slope)" if dir_sign > 0 else "(+ down-slope)"
    else:
        axA.plot([0.1, track_len-0.1], [0.3, 0.3], linewidth=3)
        def pos_to_xy(s):
            return 0.7 + s, 0.55
        axis_dir_text = "(+ right)"

    axA.text(0.05, 1.55, f"Axis {axis_dir_text}", fontsize=10)

    bx, by = pos_to_xy(0.0)
    block = Rectangle((bx - block_w/2, by - block_h/2), block_w, block_h, linewidth=1.5, edgecolor='k', facecolor='none')
    axA.add_patch(block)

    arr_app = make_arrow(axA)
    arr_fric = make_arrow(axA)
    arr_N    = make_arrow(axA)
    arr_W    = make_arrow(axA)
    status_text = axA.text(track_len*0.55, 1.30, "", fontsize=12, fontweight='bold')

    frame_dt = 1.0 / fps
    next_frame_t = 0.0
    i = 0
    # --- choose which sim index to render this frame ---
    while i < len(t)-1 and t[i] < next_frame_t:
        i += 1

        # Check for edge hit using the *simulation* position, not the clipped one
        hit_left  = (x[i] <= s_min)
        hit_right = (x[i] >= s_max)

        # Map position to screen (clipped so the block stays visible)
        s_pos = np.clip(x[i], s_min, s_max)
        bx, by = pos_to_xy(s_pos)
        block.set_xy((bx - block_w/2, by - block_h/2))

        # Unit vectors along axis and normal (same as before)
        if use_incline:
            ux, uy = np.cos(theta)*dir_sign, np.sin(theta)*dir_sign
            nx, ny = -np.sin(theta), np.cos(theta)
        else:
            ux, uy = 1.0, 0.0
            nx, ny = 0.0, 1.0

        cx, cy = bx, by

        # If we’ve reached the ledge, draw one last frame and stop
        if hit_left or hit_right:
            # Zero the dynamic arrows for a clean final frame (optional)
            update_arrow(arr_app, cx, cy, ux, uy, 0.0,       scale=0.004)
            update_arrow(arr_fric, cx, cy, ux, uy, 0.0,      scale=0.004)
            update_arrow(arr_N,   cx, cy, nx, ny, N,         scale=0.004)
            update_arrow(arr_W,   cx, cy, 0.0, -1.0, mass*g, scale=0.004)

            status_text.set_text("Reached edge — animation stopped.")
            placeholder.pyplot(figA)
            break

        # Otherwise update arrows normally and continue
        update_arrow(arr_app, cx, cy, ux, uy, F_app[i], scale=0.004)
        update_arrow(arr_fric, cx, cy, ux, uy, F_fric[i], scale=0.004)
        update_arrow(arr_N,   cx, cy, nx, ny, N,         scale=0.004)
        update_arrow(arr_W,   cx, cy, 0.0, -1.0, mass*g, scale=0.004)

        status_text.set_text(f"Regime: {'Static' if regime[i]==0 else 'Kinetic'}   "
                            f"F_app={F_app[i]:.0f} N,  F_fr={F_fric[i]:.0f} N")

        placeholder.pyplot(figA)
        next_frame_t += frame_dt
        time.sleep(max(0.0, frame_dt * 0.85))


    plt.close(figA)


# Run directly (optional)
if __name__ == "__main__":
    import streamlit.web.bootstrap as boot
    boot.run("FrictionModels.py", "", [], {})
