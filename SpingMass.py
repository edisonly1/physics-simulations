# SpringMass.py — AP Physics 1: Spring–Mass System (with Play/Pause)
from __future__ import annotations
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- physics core ----------
def euler_cromer(m, k, b, x0, v0, t_end, dt):
    n = int(np.ceil(t_end / dt)) + 1
    t = np.linspace(0, t_end, n)
    x = np.zeros(n); v = np.zeros(n)
    x[0], v[0] = x0, v0
    for i in range(n - 1):
        a = -(b / m) * v[i] - (k / m) * x[i]
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt
    return t, x, v

def energies(m, k, x, v):
    KE = 0.5 * m * v**2
    PE = 0.5 * k * x**2
    return KE, PE, KE + PE

# ---------- drawing helpers ----------
def draw_mass_spring(ax, x_now, L_track=1.2, mass_w=0.18, mass_h=0.12):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, L_track + 0.1)
    ax.set_ylim(-0.25, 0.25)
    ax.axis('off')
    ax.plot([0, L_track], [-0.15, -0.15], linewidth=3)          # Track
    ax.add_patch(plt.Rectangle((-0.03, -0.15), 0.03, 0.30, color="#999999"))  # Wall
    # Spring
    x_block_left = 0.55 + x_now - mass_w / 2
    spring_end = max(0.08, x_block_left)
    n_coils = 8
    xs = np.linspace(0.02, spring_end, n_coils * 2 + 3)
    ys = np.zeros_like(xs)
    amp = 0.05
    for i in range(1, len(xs) - 1):
        ys[i] = amp if i % 2 else -amp
    ax.plot(xs, ys, linewidth=2)
    # Block
    block_center = 0.55 + x_now
    ax.add_patch(plt.Rectangle((block_center - mass_w / 2, -mass_h / 2),
                               mass_w, mass_h, facecolor="#4e79a7"))
    ax.text(block_center, 0.12, "m", ha="center", va="bottom", fontsize=12)

def energy_bar_chart(ax, KE_now, PE_now):
    ax.clear()
    total = max(1e-9, KE_now + PE_now)
    ax.set_xlim(0, total * 1.1)
    ax.set_ylim(-0.5, 1.5)
    ax.barh([1, 0], [PE_now, KE_now], tick_label=["PEₛ", "KE"])
    ax.set_xlabel("Energy (J)")
    ax.grid(axis="x", alpha=0.3)

# ---------- session keys ----------
SS = {
    "sm_play": False,      # playing?
    "sm_loop": True,       # loop at end?
    "sm_idx": 0,           # current time index
    "sm_speed": 1.0,       # playback speed multiplier
    "sm_fps": 30,          # animation FPS
}
def _ensure_session_state():
    for k, v in SS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------- streamlit app ----------
def app():
    _ensure_session_state()

    st.title("Spring–Mass System (Hooke’s Law & Energy)")
    st.caption("m·x¨ + b·x˙ + k·x = 0   •   F = −kx   •   PEₛ = ½kx²   •   T = 2π√(m/k) (when b = 0)")

    colL, colR = st.columns([1, 2], gap="large")

    # ===== Left: controls =====
    with colL:
        st.subheader("Controls")

        m = st.slider("Mass m (kg)", 0.10, 5.00, 1.00, 0.05)
        k = st.slider("Spring constant k (N/m)", 5.0, 500.0, 50.0, 1.0)

        init_mode = st.radio("Initial condition", ["Release from rest", "Give it an initial speed"])
        A = st.slider("Initial displacement x₀ (m)", -0.40, 0.40, 0.20, 0.01)
        v0 = 0.0
        if init_mode == "Give it an initial speed":
            v0 = st.slider("Initial velocity v₀ (m/s)", -2.0, 2.0, 0.80, 0.05)

        damping_on = st.toggle("Enable damping (b > 0)", value=False)
        b = st.slider("Damping b (kg/s)", 0.0, 2.0, 0.20, 0.01, disabled=not damping_on)
        if not damping_on:
            b = 0.0

        sim_time = st.slider("Simulation length (s)", 1.0, 20.0, 8.0, 0.5)
        dt = st.select_slider("Time step dt (s)", options=[0.002, 0.005, 0.01, 0.02, 0.05], value=0.01)

        # Precompute trajectory
        t, x, v = euler_cromer(m, k, b, A, v0, sim_time, dt)
        KE, PE, TE = energies(m, k, x, v)

        # ===== Playback controls =====
        st.subheader("Playback")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.3])
        with c1:
            if not st.session_state.sm_play:
                if st.button("▶ Play", use_container_width=True):
                    st.session_state.sm_play = True
            else:
                if st.button("⏸ Pause", use_container_width=True):
                    st.session_state.sm_play = False
        with c2:
            if st.button("⟲ Reset", use_container_width=True):
                st.session_state.sm_idx = 0
        with c3:
            st.session_state.sm_loop = st.toggle("Loop", value=st.session_state.sm_loop)
        with c4:
            st.session_state.sm_speed = st.slider("Speed (×)", 0.25, 4.0, float(st.session_state.sm_speed), 0.25)
        st.session_state.sm_fps = st.slider("FPS", 10, 60, int(st.session_state.sm_fps), 5)

        # ===== Scrubber (manual) =====
        # When playing, show a read-only indicator; when paused, enable slider.
        if st.session_state.sm_play:
            i = int(np.clip(st.session_state.sm_idx, 0, len(t) - 1))
            st.progress(i / (len(t) - 1))
            st.caption(f"t = {t[i]:.3f} s  (scrubber disabled during playback)")
        else:
            t_now = st.slider("Scrub time t (s)", float(t[0]), float(t[-1]),
                              float(t[int(st.session_state.sm_idx)]), step=float(dt))
            st.session_state.sm_idx = int(np.clip(round(t_now / dt), 0, len(t) - 1))

        # Current frame
        i = int(np.clip(st.session_state.sm_idx, 0, len(t) - 1))
        x_now, v_now = float(x[i]), float(v[i])
        F_now = -k * x_now
        KE_now, PE_now = float(KE[i]), float(PE[i])

        # Readouts
        st.markdown("**Outputs (at current time)**")
        cA, cB = st.columns(2)
        with cA:
            st.metric("x (m)", f"{x_now:.3f}")
            st.metric("F = −kx (N)", f"{F_now:.3f}")
        with cB:
            st.metric("v (m/s)", f"{v_now:.3f}")
            st.metric("E_total (J)", f"{(KE_now+PE_now):.3f}")

        T = 2 * np.pi * np.sqrt(m / k)
        st.info(f"Undamped period T ≈ **{T:.3f} s**  (valid when b = 0).")

        with st.expander("Show formulas"):
            st.latex(r"F=-kx,\quad PE_s=\tfrac12 kx^2,\quad T=2\pi\sqrt{\tfrac{m}{k}}\ \ (b=0)")
            st.markdown("Integrated with **Euler–Cromer** so energy behaves well even when b>0.")

    # ===== Right: visuals =====
    with colR:
        tab1, tab2, tab3 = st.tabs(["Animation + Energy Bars", "x(t), v(t), F(t)", "Energy vs time"])

        with tab1:
            fig, (ax_anim, ax_bar) = plt.subplots(1, 2, figsize=(10, 3.2))
            draw_mass_spring(ax_anim, x_now)
            energy_bar_chart(ax_bar, KE_now, PE_now)
            fig.suptitle("Mass on a horizontal spring")
            st.pyplot(fig, clear_figure=True)

        with tab2:
            fig2 = plt.figure(figsize=(8, 3.2))
            plt.plot(t, x, label="x(t)")
            plt.plot(t, v, label="v(t)")
            plt.plot(t, -k * x, label="F(t) = -kx")
            plt.axvline(t[i], linestyle="--", alpha=0.5)
            plt.xlabel("time (s)")
            plt.legend(); plt.grid(alpha=0.3)
            st.pyplot(fig2, clear_figure=True)

        with tab3:
            fig3 = plt.figure(figsize=(8, 3.2))
            plt.plot(t, KE, label="KE")
            plt.plot(t, PE, label="PEₛ")
            plt.plot(t, KE + PE, label="Total")
            plt.axvline(t[i], linestyle="--", alpha=0.5)
            plt.xlabel("time (s)"); plt.ylabel("Energy (J)")
            plt.legend(); plt.grid(alpha=0.3)
            st.pyplot(fig3, clear_figure=True)

    # ===== Playback advance & rerun =====
    if st.session_state.sm_play:
        fps = int(st.session_state.sm_fps)
        speed = float(st.session_state.sm_speed)
        step = max(1, int(round(speed * (1.0 / fps) / dt)))  # frames to skip per tick
        new_idx = st.session_state.sm_idx + step
        if new_idx >= len(t):
            if st.session_state.sm_loop:
                new_idx = 0
            else:
                new_idx = len(t) - 1
                st.session_state.sm_play = False
        st.session_state.sm_idx = new_idx
        time.sleep(1.0 / fps)
        st.rerun()

if __name__ == "__main__":
    app()
