"""
Projectile Motion (AP Physics 1)

Features
- Sliders: v0, angle, initial height h0, duration T, time step dt, gravity g
- Outputs: time of flight, range, max height, vx, v0y
- Graphs: x–t, y–t, and x–y trajectory with overlay of saved runs
- Animation: play/pause/reset with FPS and speed controls
- Preset: one‑click classroom example
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------- Data model --------------------------- #
@dataclass
class PMRun:
    v0: float       # m/s
    angle_deg: float  # degrees
    h0: float       # m
    g: float        # m/s^2
    T: float        # s (duration used for plots/animation)
    note: str = ""

    def label(self) -> str:
        base = f"v0={self.v0:g} m/s, θ={self.angle_deg:g}°, h0={self.h0:g} m, g={self.g:g}"
        return f"{base} {f'({self.note})' if self.note else ''}".strip()


# --------------------------- Helpers --------------------------- #
def _ensure_session_state():
    st.session_state.setdefault("pm_runs", [])  # type: List[PMRun]
    st.session_state.setdefault("pm_defaults", {"v0": 25.0, "angle": 45.0, "h0": 0.0, "g": 9.81, "T": 6.0})
    st.session_state.setdefault("pm_anim_running", False)
    st.session_state.setdefault("pm_anim_idx", 0)


def kinematics(v0: float, angle_deg: float, h0: float, g: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return x(t), y(t), vx, v0y, t_flight.

    t_flight is the physical time until y returns to 0 (ground), computed analytically.
    If the trajectory never hits y=0 within the provided t-range, we still return the analytic t_flight for reference.
    """
    theta = np.deg2rad(angle_deg)
    vx = v0 * np.cos(theta)
    v0y = v0 * np.sin(theta)

    # Analytic time of flight (to ground, y=0)
    disc = v0y**2 + 2 * g * h0
    if disc < 0:
        t_flight = 0.0
    else:
        t_flight = (v0y + np.sqrt(disc)) / g  # positive root

    x = vx * t
    y = h0 + v0y * t - 0.5 * g * t**2
    return x, y, vx, v0y, t_flight


def max_height(v0: float, angle_deg: float, h0: float, g: float) -> float:
    theta = np.deg2rad(angle_deg)
    v0y = v0 * np.sin(theta)
    t_up = v0y / g if g > 0 else 0.0
    return h0 + v0y * t_up - 0.5 * g * t_up**2


# --------------------------- UI --------------------------- #
def app(data: Optional[dict] = None):
    st.title("Projectile Motion (2D)")
    _ensure_session_state()

    with st.expander("What this shows", expanded=False):
        st.markdown(
            r"""
**Purpose.** Visualize projectiles launched with speed $v_0$ at angle $\theta$ from height $h_0$ under constant gravity $g$.

**Equations used**
- $x(t) = v_0\cos\theta\; t$
- $y(t) = h_0 + v_0\sin\theta\; t - \tfrac{1}{2}gt^2$
- $t_{\text{flight}} = \dfrac{v_0\sin\theta + \sqrt{(v_0\sin\theta)^2 + 2gh_0}}{g}$
- $\text{range} = v_0\cos\theta\; t_{\text{flight}}$
            """
        )

    # --- Presets row ---
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Classroom example", help="Load: v0=20 m/s, θ=40°, h0=1.0 m, g=9.81, T=5"):
            st.session_state.v0 = 20.0
            st.session_state.angle = 40.0
            st.session_state.h0 = 1.0
            st.session_state.g = 9.81
            st.session_state.T = 5.0
            st.session_state.pm_defaults.update({"v0": 20.0, "angle": 40.0, "h0": 1.0, "g": 9.81, "T": 5.0})
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
    with c2:
        if st.button("Reset overlays"):
            st.session_state.pm_runs = []
            st.success("Cleared saved runs.")
    with c3:
        launch_mode = st.radio(
            "Launch type",
            ("From ground (h0 = 0)", "From height (h0 > 0)"),
            horizontal=True,
        )

    # --- Controls ---
    d = st.session_state.pm_defaults

    if data:  # Prefill from AI extraction if provided
        st.info("Using AI‑extracted values where available. You can still adjust sliders.")
        st.session_state.v0 = float(data.get("initial_velocity", d["v0"]))
        st.session_state.angle = float(data.get("angle", d["angle"]))
        st.session_state.h0 = float(data.get("height", d["h0"]))
        st.session_state.g = float(data.get("g", d["g"]))
        st.session_state.T = float(data.get("time", d["T"]))

    v0 = st.slider(
        "Initial speed, v0 (m/s)", 0.0, 100.0, float(st.session_state.get("v0", d["v0"])), step=0.1, key="v0"
    )
    angle = st.slider(
        "Launch angle, θ (degrees)", 0.0, 90.0, float(st.session_state.get("angle", d["angle"])), step=0.1, key="angle"
    )

    if launch_mode.startswith("From ground"):
        h0 = 0.0
    else:
        h0 = st.slider(
            "Initial height, h0 (m)", 0.0, 50.0, float(st.session_state.get("h0", d["h0"])), step=0.1, key="h0"
        )

    g = st.slider(
        "Gravity, g (m/s²)", 1.0, 20.0, float(st.session_state.get("g", d["g"])), step=0.01, key="g",
        help="Use 9.81 for Earth, ~1.62 for Moon, ~3.71 for Mars"
    )

    T = st.slider(
        "Duration, T (s)", 0.1, 60.0, float(st.session_state.get("T", d["T"])), step=0.1, key="T"
    )
    dt = st.slider(
        "Time step for plots (s)", 0.01, 0.5, 0.05, step=0.01, help="Smaller dt → smoother curves (more points)."
    )

    # Save current as default for next open
    st.session_state.pm_defaults.update({"v0": v0, "angle": angle, "h0": h0, "g": g, "T": T})

    # --- Compute ---
    t = np.arange(0.0, T + 1e-12, dt)
    x, y, vx, v0y, t_flight = kinematics(v0, angle, h0, g, t)
    rng = vx * t_flight
    h_max = max_height(v0, angle, h0, g)

    # --- Outputs ---
    st.subheader("Outputs")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Time of flight (analytic)", f"{t_flight:.3f} s")
    with colB:
        st.metric("Horizontal range (analytic)", f"{rng:.3f} m")
    with colC:
        st.metric("Max height", f"{h_max:.3f} m")

    colD, colE = st.columns(2)
    with colD:
        st.metric("Horizontal speed vx", f"{vx:.3f} m/s")
    with colE:
        st.metric("Initial vertical speed v0y", f"{v0y:.3f} m/s")

    # --- Overlay controls ---
    oc1, _ = st.columns([1, 3])
    with oc1:
        note = st.text_input("Label for this run (optional)", value="")
        if st.button("Add to overlay"):
            st.session_state.pm_runs.append(PMRun(v0=v0, angle_deg=angle, h0=h0, g=g, T=T, note=note))
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

    # x–t
    fig1, ax1 = plt.subplots()
    ax1.plot(t, x, label="Current run", linewidth=2)
    for r in st.session_state.pm_runs:
        tt = np.arange(0.0, r.T + 1e-12, dt)
        xx, yy, *_ = kinematics(r.v0, r.angle_deg, r.h0, r.g, tt)
        ax1.plot(tt, xx, linestyle="--", label=r.label())
    _styled_axes(ax1, "time t (s)", "x position (m)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig1, use_container_width=True)

    # y–t
    fig2, ax2 = plt.subplots()
    ax2.plot(t, y, label="Current run", linewidth=2)
    for r in st.session_state.pm_runs:
        tt = np.arange(0.0, r.T + 1e-12, dt)
        xx, yy, *_ = kinematics(r.v0, r.angle_deg, r.h0, r.g, tt)
        ax2.plot(tt, yy, linestyle="--", label=r.label())
    _styled_axes(ax2, "time t (s)", "y height (m)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig2, use_container_width=True)

    # Trajectory x–y
    fig3, ax3 = plt.subplots()
    ax3.plot(x, y, label="Current run", linewidth=2)
    for r in st.session_state.pm_runs:
        tt = np.arange(0.0, r.T + 1e-12, dt)
        xx, yy, *_ = kinematics(r.v0, r.angle_deg, r.h0, r.g, tt)
        ax3.plot(xx, yy, linestyle="--", label=r.label())
    # ground line at y=0
    ax3.axhline(0.0, color="black", linewidth=1)
    _styled_axes(ax3, "x (m)", "y (m)")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig3, use_container_width=True)

    # --- Animation ---
    st.subheader("Animation: 2D Flight")
    an_c1, an_c2, an_c3, an_c4 = st.columns([1, 1, 1, 2])
    with an_c1:
        if st.button("▶ Play", use_container_width=True):
            st.session_state.pm_anim_running = True
    with an_c2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.pm_anim_running = False
    with an_c3:
        if st.button("⟲ Reset", use_container_width=True):
            st.session_state.pm_anim_idx = 0
            st.session_state.pm_anim_running = False
    with an_c4:
        fps = st.slider("Playback FPS", 5, 60, 30)
        speed = st.selectbox("Speed", [0.25, 0.5, 1.0, 1.5, 2.0], index=2, help="1.0× ≈ real-time if dt ≈ 1/fps")

    # Determine nice bounds
    x_max = float(np.max(x))
    y_max = float(max(np.max(y), 0.0))
    x_pad = 0.05 * (x_max if x_max > 0 else 1.0)
    y_pad = 0.10 * (y_max if y_max > 0 else 1.0)
    xlim = (0.0, max(1.0, x_max + x_pad))
    ylim = (0.0, max(1.0, y_max + y_pad))

    # Precompute for animation timeline separate from plot dt so we always end at T
    t_anim = np.arange(0.0, T + 1e-12, dt)
    x_anim, y_anim, _, _, _ = kinematics(v0, angle, h0, g, t_anim)

    def draw_frame(idx: int):
        idx = max(0, min(idx, len(t_anim) - 1))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y, color="#cccccc", linestyle="--", label="Full path")
        ax.plot(x_anim[: idx + 1], y_anim[: idx + 1], linewidth=2, label="Path so far")
        ax.scatter([x_anim[idx]], [y_anim[idx]], s=80, zorder=3)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"t = {t_anim[idx]:.2f} s")
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig

    placeholder = st.empty()
    # Always render current frame
    placeholder.pyplot(draw_frame(st.session_state.pm_anim_idx), use_container_width=True)

    # Play loop
    if st.session_state.pm_anim_running:
        start = st.session_state.pm_anim_idx
        step = max(1, int(round(speed * (1.0 / max(dt, 1e-6)) / fps)))
        for i in range(start, len(t_anim), step):
            placeholder.pyplot(draw_frame(i), use_container_width=True)
            st.session_state.pm_anim_idx = i
            time.sleep(1.0 / fps)
            if not st.session_state.pm_anim_running:
                break
        else:
            st.session_state.pm_anim_running = False


if __name__ == "__main__":
    app()
