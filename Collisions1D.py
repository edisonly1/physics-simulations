# Collisions1D.py  — animated, overlap-safe 1D collisions
from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Physics helpers ----------
def final_velocities(m1: float, m2: float, u1: float, u2: float, mode: str) -> tuple[float, float]:
    if mode == "Elastic":
        v1 = ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)
        v2 = ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)
        return float(v1), float(v2)
    elif mode == "Perfectly Inelastic":
        v = (m1 * u1 + m2 * u2) / (m1 + m2)
        return float(v), float(v)
    else:
        raise ValueError("mode must be 'Elastic' or 'Perfectly Inelastic'")

def step_no_overlap(x1, x2, v1, v2, m1, m2, mode, w, dt):
    """
    Advance by dt with exact contact resolution so carts never overlap.
    Carts are centered at x1, x2 with identical width w (1D on a track).
    Returns (x1, x2, v1, v2, collided_this_step, t_hit_in_step_or_None)
    """
    # Relative velocity (cart1 chasing cart2 from left)
    vrel = v1 - v2

    # Distance between facing edges (right edge of 1 to left edge of 2)
    gap = (x2 - w/2) - (x1 + w/2)

    # If they are approaching and will meet within this dt, find hit time
    if vrel > 0 and gap >= 0:
        t_hit = gap / vrel
        if 0 <= t_hit <= dt:
            # Move to contact exactly
            x1 += v1 * t_hit
            x2 += v2 * t_hit

            # Compute post-collision velocities
            v1p, v2p = final_velocities(m1, m2, v1, v2, mode)

            # Advance remainder of the step with new velocities
            rem = dt - t_hit
            x1 += v1p * rem
            x2 += v2p * rem

            if mode == "Perfectly Inelastic":
                # Keep them just touching after sticking
                # Put cart2 immediately to the right of cart1
                mid = (x1 + x2) / 2.0
                x1 = mid - w/2
                x2 = mid + w/2

            return x1, x2, v1p, v2p, True, t_hit

    # Otherwise, simple uniform motion
    x1 += v1 * dt
    x2 += v2 * dt
    return x1, x2, v1, v2, False, None

# ---------- Drawing ----------
def draw_track_and_carts(x1, x2, L, w=0.9, h=0.45) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot([0, L], [0, 0], linewidth=6, alpha=0.25)  # track

    for x, color in zip([x1, x2], ["tab:blue", "tab:orange"]):
        left = x - w/2
        rect = plt.Rectangle((left, 0.05), w, h, ec="black", fc=color, alpha=0.95)
        ax.add_patch(rect)
        ax.plot([x], [0.05 + h + 0.02], marker="v", ms=6)

    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Position (m)")
    ax.get_yaxis().set_visible(False)
    ax.set_title("1D Collision Animation (Play to watch)")
    fig.tight_layout()
    return fig

# ---------- Streamlit App ----------
def app():
    st.title("Elastic & Inelastic Collisions (1D)")

    with st.expander("Show core equations"):
        st.latex(r"m_1 u_1 + m_2 u_2 = m_1 v_1 + m_2 v_2")
        st.latex(r"\text{Elastic only: } KE_i = KE_f")
        st.latex(r"v_1=\frac{(m_1-m_2)u_1+2m_2u_2}{m_1+m_2},\quad"
                 r"v_2=\frac{(m_2-m_1)u_2+2m_1u_1}{m_1+m_2}")
        st.latex(r"v_{\text{stick}}=\frac{m_1 u_1 + m_2 u_2}{m_1+m_2}")

    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader("Inputs")
        m1 = st.slider("Mass m₁ (kg)", 0.1, 10.0, 1.0, 0.1)
        m2 = st.slider("Mass m₂ (kg)", 0.1, 10.0, 1.0, 0.1)
        u1 = st.slider("Initial velocity u₁ (m/s)", -10.0, 10.0, 4.0, 0.1)
        u2 = st.slider("Initial velocity u₂ (m/s)", -10.0, 10.0, 0.0, 0.1)
        mode = st.radio("Collision type", ["Elastic", "Perfectly Inelastic"], horizontal=True)

        st.markdown("**Geometry & Animation**")
        L = st.slider("Track length (m)", 6.0, 20.0, 12.0, 0.5)
        CART_W = 1.0
        sep_min = CART_W + 0.2
        separation = st.slider("Initial separation (m) — center₂ minus center₁",
                               sep_min, L - 1.2, min(6.0, L - 1.6), 0.1)
        duration = st.slider("Run time (s)", 1.0, 15.0, 6.0, 0.1)
        speed = st.slider("Playback speed", 0.25, 2.0, 1.0, 0.05)

    with colR:
        st.subheader("Status / Outputs")
        v1f, v2f = final_velocities(m1, m2, u1, u2, mode)
        KEi = 0.5 * m1 * u1**2 + 0.5 * m2 * u2**2
        KEf = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        pi, pf = m1*u1 + m2*u2, m1*v1f + m2*v2f

        st.write(f"**Predicted final velocities** → v₁ = `{v1f:.3f}` m/s, v₂ = `{v2f:.3f}` m/s")
        st.write(f"**Momentum**: pᵢ = `{pi:.3f}`, p_f = `{pf:.3f}` (Δp = `{pf-pi:+.2e}`)")
        st.write(f"**Kinetic Energy**: KEᵢ = `{KEi:.3f}` J, KE_f = `{KEf:.3f}` J "
                 f"(ΔKE = `{KEf-KEi:+.3f}` J)")

    # --- Animation state ---
    # Reset when key inputs change (keeps things sane)
    keys = ("m1", "m2", "u1", "u2", "mode", "L", "sep", "dur")
    cur_sig = (m1, m2, u1, u2, mode, L, separation, duration)
    if "coll_sig" not in st.session_state or st.session_state.coll_sig != cur_sig:
        st.session_state.coll_sig = cur_sig
        st.session_state.t = 0.0
        st.session_state.x1 = 0.8
        st.session_state.x2 = min(st.session_state.x1 + separation, L - 0.8)
        st.session_state.v1 = u1
        st.session_state.v2 = u2
        st.session_state.playing = False

    x1 = st.session_state.x1
    x2 = st.session_state.x2
    v1 = st.session_state.v1
    v2 = st.session_state.v2
    t  = st.session_state.t

    # Controls
    play_col, reset_col, spacer = st.columns([0.25, 0.25, 0.5])
    if play_col.button("▶Play" if not st.session_state.playing else "⏸ Pause"):
        st.session_state.playing = not st.session_state.playing
    if reset_col.button("Reset"):
        st.session_state.t = 0.0
        st.session_state.x1 = 0.8
        st.session_state.x2 = min(st.session_state.x1 + separation, L - 0.8)
        st.session_state.v1 = u1
        st.session_state.v2 = u2
        st.session_state.playing = False
        x1, x2, v1, v2, t = st.session_state.x1, st.session_state.x2, st.session_state.v1, st.session_state.v2, st.session_state.t

    # Placeholder for live frames
    frame = st.empty()

    # One draw now (so you see something when paused)
    fig = draw_track_and_carts(x1, x2, L, w=CART_W)
    frame.pyplot(fig, use_container_width=True)

    # --- Run animation loop ---
    target_fps = 60.0
    base_dt = 1.0 / target_fps
    max_wall = duration

    # To avoid long-running sessions, step only while playing and under duration
    while st.session_state.playing and st.session_state.t < max_wall:
        dt = base_dt * float(speed)
        # Step with overlap-safe contact handling
        x1, x2, v1, v2, collided, _ = step_no_overlap(
            x1, x2, v1, v2, m1, m2, mode, CART_W, dt
        )

        # Keep carts on track (simple clamp)
        x1 = float(np.clip(x1, 0.8, L - 0.8))
        x2 = float(np.clip(x2, 0.8, L - 0.8))

        # Save state
        st.session_state.x1, st.session_state.x2 = x1, x2
        st.session_state.v1, st.session_state.v2 = v1, v2
        st.session_state.t += dt

        # Draw frame
        fig = draw_track_and_carts(x1, x2, L, w=CART_W)
        frame.pyplot(fig, use_container_width=True)

        # Real-time feel
        time.sleep(base_dt * 0.85)


# Standalone
if __name__ == "__main__":
    st.set_page_config(page_title="1D Collisions (Animated)", layout="wide")
    app()
