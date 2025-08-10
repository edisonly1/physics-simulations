"""
Projectile Motion (AP Physics 1)

Features
- Sliders: v0, angle, initial height h0, duration T, time step dt, gravity g
- Outputs: time of flight, range, max height, vx, v0y
- Graphs: x–t, y–t, and x–y trajectory with overlay of saved runs (all stop exactly at y=0)
- Animation: play/pause/reset with FPS and speed controls (stops exactly at y=0)
- Preset: one‑click classroom example
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import FancyArrowPatch

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


def time_of_flight(v0: float, angle_deg: float, h0: float, g: float) -> float:
    """Analytic touchdown time for y=0 (ground). Returns 0 if no real root."""
    theta = np.deg2rad(angle_deg)
    v0y = v0 * np.sin(theta)
    disc = v0y**2 + 2 * g * h0
    if disc < 0 or g <= 0:
        return 0.0
    return (v0y + np.sqrt(disc)) / g  # positive root


def time_grid(t_end: float, dt: float) -> np.ndarray:
    times = list(np.arange(0.0, t_end, dt))
    if not np.isclose(times[-1], t_end):
        times.append(t_end)
    return np.array(times)


def kinematics(v0: float, angle_deg: float, h0: float, g: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return x(t), y(t), vx, v0y for given timeline t."""
    theta = np.deg2rad(angle_deg)
    vx = v0 * np.cos(theta)
    v0y = v0 * np.sin(theta)
    x = vx * t
    y = h0 + v0y * t - 0.5 * g * t**2
    return x, y, vx, v0y


def max_height(v0: float, angle_deg: float, h0: float, g: float) -> float:
    theta = np.deg2rad(angle_deg)
    v0y = v0 * np.sin(theta)
    t_up = v0y / g if g > 0 else 0.0
    return h0 + v0y * t_up - 0.5 * g * t_up**2

def sample_path(vx: float, v0y: float, h0: float, g: float, t0: float, t1: float, n: int):
    t_s = np.linspace(t0, max(t1, t0), max(2, n))
    x_s = vx * t_s
    y_s = h0 + v0y * t_s - 0.5 * g * t_s**2
    return t_s, x_s, y_s

# --------------------------- UI --------------------------- #
def app(data: Optional[dict] = None):
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
        "Launch angle, θ (degrees)", 0.0, 90.0, float(st.session_state.get("angle", d["angle"])) , step=0.1, key="angle"
    )

    if launch_mode.startswith("From ground"):
        h0 = 0.0
    else:
        h0 = st.slider(
            "Initial height, h0 (m)", 0.0, 50.0, float(st.session_state.get("h0", d["h0"])) , step=0.1, key="h0"
        )

    g = st.slider(
        "Gravity, g (m/s²)", 1.0, 20.0, float(st.session_state.get("g", d["g"])) , step=0.01, key="g",
        help="Use 9.81 for Earth, ~1.62 for Moon, ~3.71 for Mars"
    )

    T = st.slider(
        "Duration, T (s)", 0.1, 60.0, float(st.session_state.get("T", d["T"])) , step=0.1, key="T"
    )
    dt = 0.05  # fixed plot timestep (s); no slider


    # Save current as default for next open
    st.session_state.pm_defaults.update({"v0": v0, "angle": angle, "h0": h0, "g": g, "T": T})

    # --- Compute (trim at touchdown and ensure exact landing sample) ---
    t_flight = time_of_flight(v0, angle, h0, g)
    t_end = min(T, t_flight) if t_flight > 0 else T
    t = time_grid(t_end, dt)

    x, y, vx, v0y = kinematics(v0, angle, h0, g, t)
    # snap last point exactly to ground contact
    x[-1] = vx * t_end
    y[-1] = 0.0

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
        tf_r = time_of_flight(r.v0, r.angle_deg, r.h0, r.g)
        t_end_r = min(r.T, tf_r) if tf_r > 0 else r.T
        tt = time_grid(t_end_r, dt)
        xx, yy, vxx, _ = kinematics(r.v0, r.angle_deg, r.h0, r.g, tt)
        xx[-1] = vxx * t_end_r
        ax1.plot(tt, xx, linestyle="--", label=r.label())
    _styled_axes(ax1, "time t (s)", "x position (m)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig1, use_container_width=True)

    # y–t
    fig2, ax2 = plt.subplots()
    ax2.plot(t, y, label="Current run", linewidth=2)
    for r in st.session_state.pm_runs:
        tf_r = time_of_flight(r.v0, r.angle_deg, r.h0, r.g)
        t_end_r = min(r.T, tf_r) if tf_r > 0 else r.T
        tt = time_grid(t_end_r, dt)
        xx, yy, vxx, _ = kinematics(r.v0, r.angle_deg, r.h0, r.g, tt)
        yy[-1] = 0.0
        ax2.plot(tt, yy, linestyle="--", label=r.label())
    _styled_axes(ax2, "time t (s)", "y height (m)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig2, use_container_width=True)

    # Trajectory x–y
    fig3, ax3 = plt.subplots()
    ax3.plot(x, y, label="Current run", linewidth=2)
    for r in st.session_state.pm_runs:
        tf_r = time_of_flight(r.v0, r.angle_deg, r.h0, r.g)
        t_end_r = min(r.T, tf_r) if tf_r > 0 else r.T
        tt = time_grid(t_end_r, dt)
        xx, yy, vxx, _ = kinematics(r.v0, r.angle_deg, r.h0, r.g, tt)
        xx[-1] = vxx * t_end_r
        yy[-1] = 0.0
        ax3.plot(xx, yy, linestyle="--", label=r.label())
    # ground line at y=0
    ax3.axhline(0.0, color="black", linewidth=1)
    _styled_axes(ax3, "x (m)", "y (m)")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    st.pyplot(fig3, use_container_width=True)

 
# --- Animation (continuous time; independent of dt) ---
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
            st.session_state.pm_anim_idx = 0         # legacy, harmless
            st.session_state.pm_anim_t = 0.0         # continuous sim time
            st.session_state.pm_anim_running = False
    with an_c4:
        fps = 60
        speed = st.selectbox(
            "Speed",
            [0.25, 0.5, 1.0, 1.5, 2.0],
            index=2,
            help="Sim time per frame ≈ speed / fps (independent of dt)."
        )
        show_vec = st.checkbox("Show velocity vector", value=True)
        # Optional: if you want a size knob too, uncomment next line
        # vec_scale = st.slider("Vector scale", 0.05, 0.30, 0.12, 0.01, help="Arrow length as a fraction of axis range")
        vec_scale = 0.12  # fixed scale: ~12% of max axis span
        show_components = st.checkbox("Show vx/vy components", value=False)
        # then call:
        fig0 = draw_time(st.session_state.pm_anim_t, show_vec=show_vec, show_components=show_components)


    # Ensure state var exists
    if "pm_anim_t" not in st.session_state:
        st.session_state.pm_anim_t = 0.0

    # Precompute fixed-resolution full path (independent of dt)
    N_FULL = 500
    t_full, x_full, y_full = sample_path(vx, v0y, h0, g, 0.0, t_end, N_FULL)
    x_full[-1], y_full[-1] = vx * t_end, 0.0  # snap touchdown

    # Fixed axes so the view doesn't jump
    x_max = float(np.max(x_full))
    y_max = float(max(np.max(y_full), 0.0))
    x_pad = 0.05 * (x_max if x_max > 0 else 1.0)
    y_pad = 0.10 * (y_max if y_max > 0 else 1.0)
    xlim = (0.0, max(1.0, x_max + x_pad))
    ylim = (0.0, max(1.0, y_max + y_pad))

    def draw_time(t_now: float, show_vec=True, show_components=False, vec_scale=0.12):
        """
        Draw frame at arbitrary simulation time t_now (continuous).
        show_vec: draw net velocity vector
        show_components: draw vx and vy components as separate arrows
        vec_scale: arrow length as a fraction of the larger axis span
        """
        t_now = float(np.clip(t_now, 0.0, t_end))

        # Path-so-far (fixed resolution; independent of dt)
        frac = 0.0 if t_end <= 0 else (t_now / t_end)
        n_so_far = max(2, int(round(N_FULL * max(0.0, min(1.0, frac)))))
        _, x_sf, y_sf = sample_path(vx, v0y, h0, g, 0.0, t_now, n_so_far)

        # Current position & instantaneous velocity
        x_now = vx * t_now
        y_now = h0 + v0y * t_now - 0.5 * g * t_now**2
        vx_now = vx
        vy_now = v0y - g * t_now
        speed_mag = max(1e-9, np.hypot(vx_now, vy_now))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x_full, y_full, color="#cccccc", linestyle="--", label="Full path")
        ax.plot(x_sf, y_sf, linewidth=2, label="Path so far")
        ax.scatter([x_now], [y_now], s=80, zorder=3)
        ax.axhline(0.0, color="black", linewidth=1)

        # fixed axes you computed earlier
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"t = {t_now:.3f} s")
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ---- Velocity arrows ----
        if show_vec or show_components:
            span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
            L = vec_scale * span   # base arrow length

            # Net velocity
            if show_vec:
                dx = (vx_now / speed_mag) * L
                dy = (vy_now / speed_mag) * L
                ax.add_patch(FancyArrowPatch(
                    (x_now, y_now), (x_now + dx, y_now + dy),
                    arrowstyle="->", mutation_scale=18, linewidth=2, color="tab:green", zorder=4
                ))
                ax.text(x_now + dx, y_now + dy, " v", fontsize=10, color="tab:green", va="bottom", ha="left")

            # Components (optional)
            if show_components:
                # vx component (horizontal)
                sign_x = 1.0 if vx_now >= 0 else -1.0
                ax.add_patch(FancyArrowPatch(
                    (x_now, y_now), (x_now + sign_x * L, y_now),
                    arrowstyle="->", mutation_scale=16, linewidth=1.8, color="tab:blue", zorder=4
                ))
                ax.text(x_now + sign_x * L, y_now, " vx", fontsize=9, color="tab:blue", va="bottom", ha="left")

                # vy component (vertical)
                sign_y = 1.0 if vy_now >= 0 else -1.0
                ax.add_patch(FancyArrowPatch(
                    (x_now, y_now), (x_now, y_now + sign_y * L),
                    arrowstyle="->", mutation_scale=16, linewidth=1.8, color="tab:red", zorder=4
                ))
                ax.text(x_now, y_now + sign_y * L, " vy", fontsize=9, color="tab:red", va="bottom", ha="left")

        return fig

    placeholder = st.empty()
    fig0 = draw_time(st.session_state.pm_anim_t)
    placeholder.pyplot(fig0, use_container_width=True)
    plt.close(fig0)

    # Advance time by speed/fps per frame (no dt dependence)
    if st.session_state.pm_anim_running:
        dt_frame = float(speed) / float(fps)
        while st.session_state.pm_anim_running and st.session_state.pm_anim_t < t_end - 1e-9:
            st.session_state.pm_anim_t = min(t_end, st.session_state.pm_anim_t + dt_frame)
            figi = draw_time(st.session_state.pm_anim_t)
            placeholder.pyplot(figi, use_container_width=True)
            plt.close(figi)
            time.sleep(1.0 / fps)
        else:
            st.session_state.pm_anim_running = False


if __name__ == "__main__":
    app()
