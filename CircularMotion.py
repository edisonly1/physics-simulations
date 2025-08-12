# UniformCircularMotion.py — with legend (no on-figure text labels)

from __future__ import annotations
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- vector helpers ----------
def unit_radial_in(theta: float) -> np.ndarray:
    return np.array([-np.cos(theta), -np.sin(theta)])

def unit_tangent(theta: float) -> np.ndarray:
    return np.array([-np.sin(theta),  np.cos(theta)])

def draw_arrow(ax, origin, vec, *, scale=1.0, width=0.03, alpha=1.0, color="#000"):
    if np.allclose(vec, 0):
        return
    ox, oy = origin
    vx, vy = (vec * scale).astype(float)
    ax.arrow(
        ox, oy, vx, vy,
        length_includes_head=True,
        head_width=width*2.2,
        head_length=width*3.2,
        linewidth=2.4,
        alpha=alpha,
        color=color,
        zorder=3
    )

def value_box(label, value, unit):
    c = st.container()
    c.markdown(f"**{label}**")
    c.markdown(f"<div style='font-size:1.4rem;'>{value:.3g} {unit}</div>", unsafe_allow_html=True)

# consistent colors for legend
COL = {
    "v":   "#1f77b4",  # blue
    "a_c": "#d62728",  # red
    "T":   "#2ca02c",  # green
    "f_s": "#9467bd",  # purple
    "mg":  "#000000",  # black
}

# ---------- physics ----------
def compute_ucm(m, r, v, g, context, theta, mu_s):
    omega = v / r
    ac = v**2 / r
    Fc = m * ac
    Tperiod = 2*np.pi*r / v if v > 0 else np.inf
    forces = {}

    if context == "Tension (horizontal circle)":
        forces["Tension"] = Fc

    elif context == "Flat turn (friction provides Fc)":
        forces["Friction (required)"] = Fc
        forces["μ_s (min)"] = v**2/(r*g)

    elif context == "Vertical circle (tension + gravity)":
        # radial-inward: mv^2/r = T + mg sinθ
        T = m*v**2/r - m*g*np.sin(theta)
        forces["Tension"] = T
        forces["Gravity (inward component)"] = m*g*np.sin(theta)

    return {"ac": ac, "Fc": Fc, "omega": omega, "Tperiod": Tperiod, "forces": forces}

# ---------- plotting ----------
def plot_scene(r, theta, show_v, show_a, show_forces, show_legend, legend_loc,
               physics, context, g, m, mu_s, vec_scale):

    x = r*np.cos(theta); y = r*np.sin(theta)
    pos = np.array([x, y])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    tt = np.linspace(0, 2*np.pi, 400)
    ax.plot(r*np.cos(tt), r*np.sin(tt), linestyle="--", linewidth=1, zorder=1)
    ax.scatter([x], [y], s=140, zorder=2)

    base = max(r, 1.0)
    scale = r * vec_scale / base

    t_hat = unit_tangent(theta)
    r_in  = unit_radial_in(theta)

    # legend proxy handles (avoid duplicates)
    handles = []
    added = set()
    def add_handle(name, color):
        if name not in added:
            handles.append(Line2D([0],[0], color=color, linewidth=3, label=name))
            added.add(name)

    if show_v:
        draw_arrow(ax, pos, t_hat, scale=scale, color=COL["v"])
        add_handle("v (velocity)", COL["v"])

    if show_a:
        draw_arrow(ax, pos, r_in, scale=scale, color=COL["a_c"])
        add_handle("a_c (centripetal accel.)", COL["a_c"])

    if show_forces:
        if context in ["Tension (horizontal circle)", "Vertical circle (tension + gravity)"]:
            T = physics["forces"].get("Tension", 0.0)
            if T > 0:
                draw_arrow(ax, pos, r_in, scale=scale, color=COL["T"])
                add_handle("T (tension)", COL["T"])
            else:
                ax.text(x, y-0.18*r, "T=0 (slack)", ha="center", fontsize=10, zorder=4)
        if context == "Vertical circle (tension + gravity)":
            draw_arrow(ax, pos, np.array([0.0, -1.0]), scale=scale, color=COL["mg"])
            add_handle("mg (weight)", COL["mg"])
        if context == "Flat turn (friction provides Fc)":
            draw_arrow(ax, pos, r_in, scale=scale, color=COL["f_s"])
            add_handle("f_s (static friction)", COL["f_s"])

    if show_legend and handles:
        ax.legend(handles=handles, loc=legend_loc, frameon=True)

    L = 1.25*r
    ax.set_aspect("equal", "box")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Uniform Circular Motion")
    ax.grid(True, alpha=0.3)
    return fig

# ---------- Streamlit app ----------
def app():
    st.title("Uniform Circular Motion")

    with st.sidebar:
        st.subheader("Controls")
        m = st.slider("Mass m (kg)", 0.1, 10.0, 1.0, 0.1)
        r = st.slider("Radius r (m)", 0.2, 10.0, 2.0, 0.1)
        v = st.slider("Speed v (m/s)", 0.1, 30.0, 6.0, 0.1)
        g = st.slider("g (m/s²)", 8.0, 12.0, 9.81, 0.01)

        context = st.selectbox(
            "Force context",
            ["Kinematics only",
             "Tension (horizontal circle)",
             "Flat turn (friction provides Fc)",
             "Vertical circle (tension + gravity)"]
        )

        mu_s = 0.0; theta_deg = 0.0
        if context == "Flat turn (friction provides Fc)":
            mu_s = st.slider("Available μ_s", 0.0, 1.5, 0.5, 0.01)
        if context == "Vertical circle (tension + gravity)":
            theta_deg = st.slider("Position angle θ (deg, 0° at +x, CCW)", 0, 360, 60, 1)

        st.markdown("---")
        st.subheader("Vectors")
        show_v = st.checkbox("Show velocity →", True)
        show_a = st.checkbox("Show centripetal acceleration →", True)
        show_F = st.checkbox("Show forces →", True)
        vec_scale = st.slider("Vector length (relative)", 0.25, 1.20, 0.60, 0.01)

        st.markdown("---")
        st.subheader("Legend")
        show_legend = st.checkbox("Show legend", True)
        legend_loc = st.selectbox(
            "Legend position",
            ["best","upper right","upper left","lower left","lower right","right",
             "center left","center right","lower center","upper center","center"],
            index=0
        )

        st.markdown("---")
        st.subheader("Animation")
        period = compute_ucm(m, r, v, g, context, np.deg2rad(theta_deg), mu_s)["Tperiod"]
        if "ucm_t" not in st.session_state:
            st.session_state.ucm_t = 0.0

        st.session_state.ucm_t = st.slider(
            "t (s)",
            0.0,
            float(max(0.5, period if np.isfinite(period) else 10.0)),
            float(st.session_state.ucm_t),
            0.01,
            key="slider_t_ucm"
        )

        colA, colB = st.columns(2)
        with colA:
            play = st.button("▶ Play 1 cycle", use_container_width=True)
        with colB:
            reset = st.button("⟲ Reset", use_container_width=True)
        if reset:
            st.session_state.ucm_t = 0.0

    # current frame physics
    t = float(st.session_state.ucm_t)
    theta = (v/r)*t + (np.deg2rad(theta_deg) if context == "Vertical circle (tension + gravity)" else 0.0)
    physics = compute_ucm(m, r, v, g, context, theta, mu_s)

    left, right = st.columns([3, 2], vertical_alignment="top")

    with left:
        fig = plot_scene(r, theta, show_v, show_a, show_F, show_legend, legend_loc,
                         physics, context, g, m, mu_s, vec_scale)
        ph = st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Outputs")
        a1, a2 = st.columns(2)
        with a1: value_box("Centripetal acceleration, aₙ = v²/r", physics["ac"], "m/s²")
        with a2: value_box("Centripetal force, Fₙ = m v²/r", physics["Fc"], "N")

        b1, b2 = st.columns(2)
        with b1: value_box("Angular speed, ω = v/r", physics["omega"], "rad/s")
        with b2: value_box("Period, T = 2πr/v", physics["Tperiod"], "s")

        if context == "Flat turn (friction provides Fc)":
            mu_min = physics["forces"]["μ_s (min)"]
            st.markdown(f"**μₛ (min) needed:** {mu_min:.3g}")
            st.success("Given μₛ is sufficient (no skidding)." if mu_s >= mu_min
                       else "Given μₛ is NOT sufficient → skids outward.")

        if context in ["Tension (horizontal circle)", "Vertical circle (tension + gravity)"]:
            T = physics["forces"].get("Tension", None)
            if T is not None:
                st.markdown(f"**Tension:** {T:.3g} N" if T >= 0
                            else ":warning: String would go slack (T ≤ 0).")

        with st.expander("Show formulas"):
            st.latex(r"a_c=\frac{v^2}{r}\qquad F_c=m\frac{v^2}{r}\qquad \omega=\frac{v}{r}\qquad T=\frac{2\pi r}{v}")
            if context == "Vertical circle (tension + gravity)":
                st.latex(r"\frac{mv^2}{r}=T+mg\sin\theta\ \Rightarrow\ T=\frac{mv^2}{r}-mg\sin\theta")
            if context == "Flat turn (friction provides Fc)":
                st.latex(r"f_s^{\text{req}}=\frac{mv^2}{r},\ \ \mu_s^{\min}=\frac{v^2}{rg}")

    # autoplay
    if play and np.isfinite(period) and period > 0:
        frames = 120
        dt = period/frames
        for i in range(frames):
            t = (i+1)*dt
            st.session_state.ucm_t = t
            theta = (v/r)*t + (np.deg2rad(theta_deg) if context == "Vertical circle (tension + gravity)" else 0.0)
            physics = compute_ucm(m, r, v, g, context, theta, mu_s)
            fig = plot_scene(r, theta, show_v, show_a, show_F, show_legend, legend_loc,
                             physics, context, g, m, mu_s, vec_scale)
            ph.pyplot(fig, clear_figure=True)
            time.sleep(0.015)
        st.session_state.ucm_t = 0.0
