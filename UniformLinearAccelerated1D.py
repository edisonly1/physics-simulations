"""
Uniform Linear & Accelerated Motion (1D)
Streamlit module for AP Physics 1

Features
- Sliders: initial velocity (u), acceleration (a), duration (T), time step (dt)
- Outputs: s(T) = uT + 1/2 aT^2,  v(T) = u + aT
- Graphs: position–time, velocity–time, acceleration–time
- NEW: Simple 1D animation of the object's motion along a track with live readouts
- Overlay multiple runs for comparison
- One‑click classroom example: car accelerates from rest at 3 m/s^2 for 5 s
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------- Data model --------------------------- #
@dataclass
class Run:
    u: float  # initial velocity (m/s)
    a: float  # acceleration (m/s^2)
    T: float  # total time (s)
    note: str = ""

    def label(self) -> str:
        base = f"u={self.u:g} m/s, a={self.a:g} m/s², T={self.T:g} s"
        return f"{base} {f'({self.note})' if self.note else ''}".strip()


# --------------------------- Helpers --------------------------- #
def kinematics(u: float, a: float, t: np.ndarray):
    s = u * t + 0.5 * a * t**2
    v = u + a * t
    a_arr = np.full_like(t, a)
    return s, v, a_arr


def _ensure_session_state():
    if "u1d_runs" not in st.session_state:
        st.session_state.u1d_runs: List[Run] = []
    if "u1d_defaults" not in st.session_state:
        st.session_state.u1d_defaults = {"u": 0.0, "a": 0.0, "T": 5.0}
    # Animation state
    st.session_state.setdefault("u1d_anim_running", False)
    st.session_state.setdefault("u1d_anim_idx", 0)


# --------------------------- UI --------------------------- #
def app():
    _ensure_session_state()

    with st.expander("What this shows", expanded=False):
        st.markdown(
            r"""
**Purpose.** Visualize straight‑line motion with constant velocity or constant acceleration. 

**Equations used**
- $s(t) = ut + \tfrac{1}{2}at^2$  
- $v(t) = u + at$  
- $a(t) = a$ (constant)

**Tip.** Use **Add to overlay** to compare different parameter sets on the same graphs.
            """
        )

    # --- Presets row ---
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Classroom example", help="Load: u=0, a=3, T=5"):
            st.session_state.u = 0.0
            st.session_state.a = 3.0
            st.session_state.T = 5.0
            st.session_state.u1d_defaults.update({"u": 0.0, "a": 3.0, "T": 5.0})
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
    with c2:
        if st.button("Reset overlays"):
            st.session_state.u1d_runs = []
            st.success("Cleared saved runs.")
    with c3:
        mode = st.radio(
            "Motion type",
            ("Constant acceleration", "Constant velocity (a = 0)"),
            horizontal=True,
        )

    # --- Controls ---
    defaults = st.session_state.u1d_defaults
    u = st.slider(
        "Initial velocity, u (m/s)",
        min_value=-50.0,
        max_value=50.0,
        value=float(st.session_state.get("u", defaults["u"])) ,
        step=0.1,
        key="u",
    )
    a = st.slider(
        "Acceleration, a (m/s²)",
        min_value=-20.0,
        max_value=20.0,
        value=float(st.session_state.get("a", defaults["a"])) ,
        step=0.1,
        key="a",
        help="Set to 0 for constant velocity.",
    )
    if mode.endswith("a = 0"):
        a = 0.0

    T = st.slider(
        "Duration, T (s)",
        min_value=0.1,
        max_value=60.0,
        value=float(st.session_state.get("T", defaults["T"])) ,
        step=0.1,
        key="T",
    )
    dt = 0.05

    # Save current as default for next open
    st.session_state.u1d_defaults.update({"u": u, "a": a, "T": T})

    # --- Compute ---
    t = np.arange(0.0, T + 1e-12, dt)
    s, v, a_arr = kinematics(u, a, t)

    # Final values
    sT = s[-1]
    vT = v[-1]

    # --- Output cards ---
    st.subheader("Outputs")
    colA, colB = st.columns(2)
    with colA:
        st.latex(r"s(T) = uT + \tfrac{1}{2} a T^2")
        st.metric(label="Displacement after T", value=f"{sT:.3f} m")
    with colB:
        st.latex(r"v(T) = u + aT")
        st.metric(label="Velocity at T", value=f"{vT:.3f} m/s")

    # --- Overlay buttons ---
    oc1, _ = st.columns([1, 3])
    with oc1:
        note = st.text_input("Label for this run (optional)", value="")
        if st.button("Add to overlay"):
            st.session_state.u1d_runs.append(Run(u=u, a=a, T=T, note=note))
            st.success("Saved current run. It will appear on the graphs below.")

    # --- Graphs ---
    st.subheader("Graphs")
    show_grid = st.checkbox("Show grid", value=True)

    def _styled_axes(ax, xlab: str, ylab: str):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if show_grid:
            ax.grid(True, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Position–time
    fig1, ax1 = plt.subplots()
    ax1.plot(t, s, label="Current run", linewidth=2)
    for r in st.session_state.u1d_runs:
        tt = np.arange(0.0, r.T + 1e-12, dt)
        ss, _, _ = kinematics(r.u, r.a, tt)
        ax1.plot(tt, ss, linestyle="--", label=r.label())
    _styled_axes(ax1, "time t (s)", "position s (m)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig1, use_container_width=True)

    # Velocity–time
    fig2, ax2 = plt.subplots()
    ax2.plot(t, v, label="Current run", linewidth=2)
    for r in st.session_state.u1d_runs:
        tt = np.arange(0.0, r.T + 1e-12, dt)
        _, vv, _ = kinematics(r.u, r.a, tt)
        ax2.plot(tt, vv, linestyle="--", label=r.label())
    _styled_axes(ax2, "time t (s)", "velocity v (m/s)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig2, use_container_width=True)

    # Acceleration–time
    fig3, ax3 = plt.subplots()
    ax3.plot(t, a_arr, label="Current run", linewidth=2)
    for r in st.session_state.u1d_runs:
        tt = np.arange(0.0, r.T + 1e-12, dt)
        _, _, aa = kinematics(r.u, r.a, tt)
        ax3.plot(tt, aa, linestyle="--", label=r.label())
    _styled_axes(ax3, "time t (s)", "acceleration a (m/s²)")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig3, use_container_width=True)

    # --- Animation ---
    st.subheader("Animation: 1D Motion Along a Track")
    an_c1, an_c2, an_c3, an_c4 = st.columns([1,1,1,2])
    with an_c1:
        if st.button("▶ Play", use_container_width=True):
            st.session_state.u1d_anim_running = True
    with an_c2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.u1d_anim_running = False
    with an_c3:
        if st.button("⟲ Reset", use_container_width=True):
            st.session_state.u1d_anim_idx = 0
            st.session_state.u1d_anim_running = False
    with an_c4:
        fps = st.slider("Playback FPS", 5, 60, 30)
        speed = st.selectbox("Speed", [0.25, 0.5, 1.0, 1.5, 2.0], index=2, help="1.0× = real-time if dt ≈ 1/fps")

    # Determine spatial bounds from the trajectory for nice framing
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    if s_min == s_max:
        s_min -= 1.0
        s_max += 1.0
    pad = 0.05 * (s_max - s_min)
    s_min -= pad
    s_max += pad

    # Draw one frame helper
    def draw_frame(idx: int):
        idx = max(0, min(idx, len(t)-1))
        s_now, v_now, a_now, t_now = s[idx], v[idx], a_arr[idx], t[idx]
        fig, ax = plt.subplots(figsize=(8, 1.8))
        ax.plot([s_min, s_max], [0, 0], linewidth=3)
        ax.scatter([s_now], [0], s=120, zorder=3)
        ax.set_xlim(s_min, s_max)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlabel("position s (m)")
        ax.set_title(f"t = {t_now:.2f} s | s = {s_now:.2f} m | v = {v_now:.2f} m/s | a = {a_now:.2f} m/s²")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return fig

    placeholder = st.empty()

    # Always render current frame (even if not playing)
    placeholder.pyplot(draw_frame(st.session_state.u1d_anim_idx), use_container_width=True)

    # Play loop (blocking until done or paused)
    if st.session_state.u1d_anim_running:
        start = st.session_state.u1d_anim_idx
        step = max(1, int(round(speed * (1.0 / max(dt, 1e-6)) / fps)))
        for i in range(start, len(t)):
            placeholder.pyplot(draw_frame(i), use_container_width=True)
            st.session_state.u1d_anim_idx = i
            time.sleep(1.0 / fps)
            if not st.session_state.u1d_anim_running:
                break
        else:
            # reached the end
            st.session_state.u1d_anim_running = False


if __name__ == "__main__":
    app()
