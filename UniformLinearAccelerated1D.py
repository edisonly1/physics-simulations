"""
Uniform Linear & Accelerated Motion (1D)
Streamlit module for AP Physics 1

Features
- Sliders: initial velocity (u), acceleration (a), duration (T), time step (dt)
- Outputs: s(T) = uT + 1/2 aT^2,  v(T) = u + aT
- Graphs: position–time, velocity–time, acceleration–time
- Overlay multiple runs for comparison
- One‑click classroom example: car accelerates from rest at 3 m/s^2 for 5 s

"""
from __future__ import annotations

import io
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
    """Return s(t), v(t), a(t) arrays for 1D constant‑acceleration motion.

    s(t) = u t + 1/2 a t^2
    v(t) = u + a t
    a(t) = constant (a)
    """
    s = u * t + 0.5 * a * t**2
    v = u + a * t
    a_arr = np.full_like(t, a)
    return s, v, a_arr


def _ensure_session_state():
    if "u1d_runs" not in st.session_state:
        st.session_state.u1d_runs: List[Run] = []
    if "u1d_defaults" not in st.session_state:
        st.session_state.u1d_defaults = {"u": 0.0, "a": 0.0, "T": 5.0}


# --------------------------- UI --------------------------- #
def app():
    st.title("Uniform Linear & Accelerated Motion (1D)")
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
    dt = st.slider(
        "Time step for plots (s)",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Smaller dt → smoother curves (more points).",
    )

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

    # CSV download for current run
    csv_buf = io.StringIO()
    np.savetxt(
        csv_buf,
        np.column_stack([t, s, v, a_arr]),
        delimiter=",",
        header="t,s,v,a",
        comments="",
    )
    st.download_button(
        "Download CSV (current run)",
        data=csv_buf.getvalue(),
        file_name="uniform_1d_run.csv",
        mime="text/csv",
    )

    # --- Overlay buttons ---
    oc1, oc2 = st.columns([1, 3])
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


