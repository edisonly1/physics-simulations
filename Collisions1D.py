# Collisions1D.py — animated 1D collisions with restitution e + Energy bar charts
from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Physics helpers ----------------
def final_velocities(m1: float, m2: float, u1: float, u2: float,
                     mode: str, e: float | None = None) -> tuple[float, float, float]:
    if mode == "Elastic":
        e_used = 1.0
    elif mode == "Perfectly Inelastic":
        e_used = 0.0
    else:
        e_used = float(np.clip(0.0 if e is None else e, 0.0, 1.0))
    v1 = (m1*u1 + m2*u2 - m2*e_used*(u1 - u2)) / (m1 + m2)
    v2 = (m1*u1 + m2*u2 + m1*e_used*(u1 - u2)) / (m1 + m2)
    return float(v1), float(v2), e_used

<<<<<<< HEAD
def build_custom_force(points_df: pd.DataFrame, total_time: float, dt: float):
    df = points_df.copy()
    df["t"] = df["t"].clip(lower=0.0, upper=max(1e-3, total_time))
    df = df.sort_values("t", kind="mergesort").drop_duplicates(subset="t")
    if df["t"].iloc[0] > 0.0: df = pd.concat([pd.DataFrame([{"t": 0.0, "F": 0.0}]), df], ignore_index=True)
    if df["t"].iloc[-1] < total_time: df = pd.concat([df, pd.DataFrame([{"t": total_time, "F": 0.0}])], ignore_index=True)
=======
def step_no_overlap(x1, x2, v1, v2, m1, m2, mode, e, w, dt):
    vrel = v1 - v2
    gap = (x2 - w/2) - (x1 + w/2)
    if vrel > 0 and gap >= 0:
        t_hit = gap / vrel
        if 0 <= t_hit <= dt:
            x1 += v1 * t_hit
            x2 += v2 * t_hit
            v1p, v2p, e_used = final_velocities(m1, m2, v1, v2, mode, e)
            rem = dt - t_hit
            x1 += v1p * rem
            x2 += v2p * rem
            if e_used == 0.0:  # stick: keep touching
                mid = (x1 + x2) / 2.0
                x1 = mid - w/2
                x2 = mid + w/2
            return x1, x2, v1p, v2p, True
    x1 += v1 * dt
    x2 += v2 * dt
    return x1, x2, v1, v2, False
>>>>>>> parent of 02353c0 (Update Collisions1D.py)

# ---------------- Drawing ----------------
def draw_track_and_carts(x1, x2, L, w=1.0, h=0.45) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot([0, L], [0, 0], linewidth=6, alpha=0.25)
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

def energy_bars(ke1_i, ke2_i, ke1_f, ke2_f) -> plt.Figure:
    """Side-by-side bar charts: KE per cart BEFORE vs AFTER."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
    labels = ["Cart 1", "Cart 2"]

<<<<<<< HEAD
    a = F / max(TOL, m)
    dv = cumtrapz(a, t)
    v = v0 + dv
    x = cumtrapz(v, t)
    p = m * v
    return {"J_total": J_total, "J_contact": J_contact, "F_avg": F_avg, "v": v, "x": x, "p": p, "a": a}

# ---------- plots ----------
def plot_force_with_area(t, F, J):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, F, lw=2, label="Force $F(t)$")
    ax.fill_between(t, 0, F, alpha=0.25, label=f"Impulse area  J = {J:.3f} N·s")
    ax.axhline(0, lw=1, color="#888"); ax.set_xlabel("time (s)"); ax.set_ylabel("force (N)")
    ax.legend(loc="best"); ax.grid(alpha=0.25); fig.tight_layout(); return fig
=======
    axs[0].bar(labels, [ke1_i, ke2_i])
    axs[0].set_title("Energy BEFORE")
    axs[0].set_ylabel("Kinetic Energy (J)")
    axs[1].bar(labels, [ke1_f, ke2_f])
    axs[1].set_title("Energy AFTER")

    # annotate bars with values
    for ax in axs:
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig
>>>>>>> parent of 02353c0 (Update Collisions1D.py)

def energy_pies(ke1_i, ke2_i, ke1_f, ke2_f) -> plt.Figure:
    """Optional: pies if you want them."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
    eps = 1e-9
    axs[0].pie([max(ke1_i, eps), max(ke2_i, eps)],
               labels=[f"Cart 1\n{ke1_i:.2f} J", f"Cart 2\n{ke2_i:.2f} J"], autopct=lambda p: f"{p:.0f}%")
    axs[0].set_title("Energy BEFORE")
    axs[1].pie([max(ke1_f, eps), max(ke2_f, eps)],
               labels=[f"Cart 1\n{ke1_f:.2f} J", f"Cart 2\n{ke2_f:.2f} J"], autopct=lambda p: f"{p:.0f}%")
    axs[1].set_title("Energy AFTER")
    fig.tight_layout()
    return fig

# ---------------- Streamlit App ----------------
def app():
    st.title("Elastic / Inelastic Collisions (1D)")

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

    # Predicted finals (for the charts)
    v1f, v2f, e_used = final_velocities(m1, m2, u1, u2,
                                        "Partially Inelastic" if mode.startswith("Partially") else mode,
                                        e_slider)

    with colR:
        st.subheader("Outputs")
        KEi = 0.5 * m1 * u1**2 + 0.5 * m2 * u2**2
        KEf = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        pi, pf = m1*u1 + m2*u2, m1*v1f + m2*v2f
        st.write(f"**e used**: `{e_used:.2f}`")
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

<<<<<<< HEAD
        # -------- Animation with Play/Pause/Reset ----------
        st.subheader("Animation: ball struck by force profile")

        t_min, t_max = float(t[0]), float(t[-1])
        step_time = float(max(dt, (t_max - t_min) / 200.0))

        # session state
        if "imp_t_now" not in st.session_state: st.session_state.imp_t_now = t_min
        if "imp_is_playing" not in st.session_state: st.session_state.imp_is_playing = False
        if "imp_speed" not in st.session_state: st.session_state.imp_speed = 1.0
        # clamp if user changed time window
        st.session_state.imp_t_now = float(np.clip(st.session_state.imp_t_now, t_min, t_max))

        c1, c2, c3 = st.columns([1, 1, 2])
        if not st.session_state.imp_is_playing:
            if c1.button("Play", use_container_width=True):
                st.session_state.imp_is_playing = True
                st.rerun()
        else:
            if c1.button("Pause", use_container_width=True):
                st.session_state.imp_is_playing = False
                st.rerun()

        if c2.button("Reset", use_container_width=True):
            st.session_state.imp_t_now = t_min
            st.session_state.imp_is_playing = False
            st.rerun()

        st.session_state.imp_speed = c3.select_slider(
            "Speed", options=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0], value=st.session_state.imp_speed
=======
    x1 = st.session_state.x1
    x2 = st.session_state.x2
    v1 = st.session_state.v1
    v2 = st.session_state.v2

    # Controls
    play_col, reset_col, _ = st.columns([0.25, 0.25, 0.5])
    if play_col.button("▶Play" if not st.session_state.playing else "Pause"):
        st.session_state.playing = not st.session_state.playing
    if reset_col.button("Reset"):
        st.session_state.t = 0.0
        st.session_state.x1 = 0.8
        st.session_state.x2 = min(st.session_state.x1 + separation, L - 0.8)
        st.session_state.v1 = u1
        st.session_state.v2 = u2
        st.session_state.playing = False
        x1 = st.session_state.x1; x2 = st.session_state.x2
        v1 = st.session_state.v1; v2 = st.session_state.v2

    frame = st.empty()
    fig = draw_track_and_carts(x1, x2, L, w=CART_W)
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
>>>>>>> parent of 02353c0 (Update Collisions1D.py)
        )
        x1 = float(np.clip(x1, 0.8, L - 0.8))
        x2 = float(np.clip(x2, 0.8, L - 0.8))
        st.session_state.x1, st.session_state.x2 = x1, x2
        st.session_state.v1, st.session_state.v2 = v1, v2
        st.session_state.t += dt
        fig = draw_track_and_carts(x1, x2, L, w=CART_W)
        frame.pyplot(fig, use_container_width=True)
        time.sleep(base_dt * 0.85)

<<<<<<< HEAD
        t_now = st.slider(
            "Scrub time",
            min_value=t_min, max_value=t_max,
            value=float(st.session_state.imp_t_now),
            step=step_time, format="%.3f", key="imp_scrubber"
        )
        st.session_state.imp_t_now = float(t_now)

        st.pyplot(draw_ball_panel(t, x, v, st.session_state.imp_t_now), use_container_width=True)

        if st.session_state.imp_is_playing:
            time.sleep(0.016)  # ~60 FPS-ish
            next_t = st.session_state.imp_t_now + st.session_state.imp_speed * step_time
            if next_t >= t_max:
                st.session_state.imp_t_now = t_max
                st.session_state.imp_is_playing = False
            else:
                st.session_state.imp_t_now = float(next_t)
            st.rerun()

        st.markdown("*Shaded area under $F(t)$ is impulse $J$.  $\\Delta v = J/m$*")

# Standalone
if __name__ == "__main__":
    st.set_page_config(page_title="Impulse–Momentum", layout="wide")
=======
    # ---------- Energy charts (Before vs After) ----------
    ke1_i, ke2_i = 0.5 * m1 * u1**2, 0.5 * m2 * u2**2
    ke1_f, ke2_f = 0.5 * m1 * v1f**2, 0.5 * m2 * v2f**2

    st.subheader("Energy comparison")
    chart_type = st.radio("Chart type", ["Bars", "Pies"], horizontal=True, index=0)
    if chart_type == "Bars":
        st.pyplot(energy_bars(ke1_i, ke2_i, ke1_f, ke2_f), use_container_width=True)
    else:
        st.pyplot(energy_pies(ke1_i, ke2_i, ke1_f, ke2_f), use_container_width=True)

    st.caption("Bars show kinetic energy carried by each cart **before** and **after** the collision. "
               "Momentum is always conserved; kinetic energy only for elastic (e=1).")

# Standalone
if __name__ == "__main__":
    st.set_page_config(page_title="1D Collisions", layout="wide")
>>>>>>> parent of 02353c0 (Update Collisions1D.py)
    app()
