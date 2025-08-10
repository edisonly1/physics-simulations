# Collisions1D.py — animated 1D collisions with coefficient of restitution e
from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Physics helpers ----------------
def final_velocities(m1: float, m2: float, u1: float, u2: float,
                     mode: str, e: float | None = None) -> tuple[float, float, float]:
    """
    Return (v1, v2, e_used) for Elastic, Perfectly Inelastic, or Partially Inelastic (with slider e).
    Uses general restitution formula: v2 - v1 = -e (u2 - u1) with momentum conservation.
    """
    if mode == "Elastic":
        e_used = 1.0
    elif mode == "Perfectly Inelastic":
        e_used = 0.0
    else:  # Partially Inelastic
        e_used = float(np.clip(0.0 if e is None else e, 0.0, 1.0))

    v1 = (m1*u1 + m2*u2 - m2*e_used*(u1 - u2)) / (m1 + m2)
    v2 = (m1*u1 + m2*u2 + m1*e_used*(u1 - u2)) / (m1 + m2)
    return float(v1), float(v2), e_used

def step_no_overlap(x1, x2, v1, v2, m1, m2, mode, e, w, dt):
    """
    Advance by dt with exact contact handling so carts never overlap.
    Carts centered at x1, x2 with identical width w.
    Returns (x1, x2, v1, v2, collided_this_step).
    """
    vrel = v1 - v2
    gap = (x2 - w/2) - (x1 + w/2)  # distance between facing edges

    # Predict contact inside this step
    if vrel > 0 and gap >= 0:
        t_hit = gap / vrel
        if 0 <= t_hit <= dt:
            # Move to first contact
            x1 += v1 * t_hit
            x2 += v2 * t_hit

            # Apply collision response using restitution
            v1p, v2p, e_used = final_velocities(m1, m2, v1, v2, mode, e)

            # Finish remainder of step with new velocities
            rem = dt - t_hit
            x1 += v1p * rem
            x2 += v2p * rem

            if e_used == 0.0:  # stick: keep them just touching
                mid = (x1 + x2) / 2.0
                x1 = mid - w/2
                x2 = mid + w/2

            return x1, x2, v1p, v2p, True

    # No contact during this step → uniform motion
    x1 += v1 * dt
    x2 += v2 * dt
    return x1, x2, v1, v2, False

# ---------------- Drawing ----------------
def draw_track_and_carts(x1, x2, L, w=1.0, h=0.45) -> plt.Figure:
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

# ---------------- Streamlit App ----------------
def app():
    st.title("Elastic / Inelastic Collisions (1D) — Animation with Restitution e")

    with st.expander("Show equations"):
        st.latex(r"p:\ m_1u_1+m_2u_2=m_1v_1+m_2v_2")
        st.latex(r"\text{Restitution: } v_2-v_1=-e\,(u_2-u_1),\quad 0\le e\le 1")
        st.latex(r"e=1 \Rightarrow \text{elastic},\ e=0 \Rightarrow \text{perfectly inelastic}")

    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader("Inputs")
        m1 = st.slider("Mass m₁ (kg)", 0.1, 10.0, 1.0, 0.1)
        m2 = st.slider("Mass m₂ (kg)", 0.1, 10.0, 1.0, 0.1)
        u1 = st.slider("Initial velocity u₁ (m/s)", -10.0, 10.0, 4.0, 0.1)
        u2 = st.slider("Initial velocity u₂ (m/s)", -10.0, 10.0, 0.0, 0.1)
        mode = st.radio(
            "Collision type",
            ["Elastic", "Perfectly Inelastic", "Partially Inelastic (choose e)"],
            horizontal=False,
        )
        e_slider = None
        if mode.startswith("Partially"):
            e_slider = st.slider("Coefficient of restitution e", 0.0, 1.0, 0.6, 0.01)

        st.markdown("**Geometry & Animation**")
        L = st.slider("Track length (m)", 6.0, 20.0, 12.0, 0.5)
        CART_W = 1.0
        sep_min = CART_W + 0.2
        separation = st.slider("Initial separation (m) — center₂ minus center₁",
                               sep_min, L - 1.2, min(6.0, L - 1.6), 0.1)
        duration = st.slider("Run time (s)", 1.0, 15.0, 6.0, 0.1)
        speed = st.slider("Playback speed", 0.25, 2.0, 1.0, 0.05)

    # Predicted finals (using chosen e)
    v1f, v2f, e_used = final_velocities(m1, m2, u1, u2,
                                        "Partially Inelastic" if mode.startswith("Partially") else mode,
                                        e_slider)

    with colR:
        st.subheader("Outputs")
        KEi = 0.5 * m1 * u1**2 + 0.5 * m2 * u2**2
        KEf = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        pi, pf = m1*u1 + m2*u2, m1*v1f + m2*v2f

        st.write(f"**e used**: `{e_used:.2f}`  "
                 f"(Elastic → 1.00, Perfectly Inelastic → 0.00)")
        st.write(f"**Final velocities** → v₁ = `{v1f:.3f}` m/s,  v₂ = `{v2f:.3f}` m/s")
        st.write(f"**Momentum**: pᵢ = `{pi:.3f}`, p_f = `{pf:.3f}` (Δp = `{pf-pi:+.2e}`)")
        dKE = KEf - KEi
        pct = (100*dKE/KEi) if KEi > 1e-12 else 0.0
        st.write(f"**Kinetic Energy**: KEᵢ = `{KEi:.3f}` J, KE_f = `{KEf:.3f}` J "
                 f"(ΔKE = `{dKE:+.3f}` J, {pct:+.1f}% change)")

    # ---------- Animation state ----------
    sig = (m1, m2, u1, u2, e_used, mode, L, separation, duration)
    if "coll_sig" not in st.session_state or st.session_state.coll_sig != sig:
        st.session_state.coll_sig = sig
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
    play_col, reset_col, _ = st.columns([0.25, 0.25, 0.5])
    if play_col.button("▶️ Play" if not st.session_state.playing else "⏸ Pause"):
        st.session_state.playing = not st.session_state.playing
    if reset_col.button("↺ Reset"):
        st.session_state.t = 0.0
        st.session_state.x1 = 0.8
        st.session_state.x2 = min(st.session_state.x1 + separation, L - 0.8)
        st.session_state.v1 = u1
        st.session_state.v2 = u2
        st.session_state.playing = False
        x1 = st.session_state.x1; x2 = st.session_state.x2
        v1 = st.session_state.v1; v2 = st.session_state.v2; t = st.session_state.t

    frame = st.empty()
    fig = draw_track_and_carts(x1, x2, L, w=CART_W)  # initial draw
    frame.pyplot(fig, use_container_width=True)

    # Run loop
    target_fps = 60.0
    base_dt = 1.0 / target_fps
    while st.session_state.playing and st.session_state.t < duration:
        dt = base_dt * float(speed)
        x1, x2, v1, v2, _ = step_no_overlap(
            x1, x2, v1, v2, m1, m2,
            "Partially Inelastic" if mode.startswith("Partially") else mode,
            e_slider, CART_W, dt
        )
        # keep on track
        x1 = float(np.clip(x1, 0.8, L - 0.8))
        x2 = float(np.clip(x2, 0.8, L - 0.8))

        st.session_state.x1, st.session_state.x2 = x1, x2
        st.session_state.v1, st.session_state.v2 = v1, v2
        st.session_state.t += dt

        fig = draw_track_and_carts(x1, x2, L, w=CART_W)
        frame.pyplot(fig, use_container_width=True)
        time.sleep(base_dt * 0.85)

    st.caption("Contact is resolved at the exact hit time each frame, so blocks never overlap. "
               "Choose **Partially Inelastic** to explore any 0 ≤ e ≤ 1.")
    
# Standalone run
if __name__ == "__main__":
    st.set_page_config(page_title="1D Collisions (e-slider)", layout="wide")
    app()
