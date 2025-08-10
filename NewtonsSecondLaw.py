# NewtonsSecondLaw.py
# Streamlit module: Newton's 2nd Law (1D/2D) — FBD + constant-acceleration motion
# Requirements: streamlit, numpy, matplotlib

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

# -----------------------------
# Utilities
# -----------------------------
DEG2RAD = math.pi / 180.0
G = 9.81  # m/s^2

@dataclass
class Force:
    name: str
    fx: float
    fy: float
    show: bool = True

def polar_to_cartesian(F: float, theta_deg: float) -> Tuple[float, float]:
    th = theta_deg * DEG2RAD
    return F * math.cos(th), F * math.sin(th)

def solve_time_to_y0(y0: float, vy0: float, ay: float) -> float | None:
    """
    Returns smallest t>0 such that y(t)=0 for motion y = y0 + vy0 t + 0.5 ay t^2.
    If no positive root, returns None.
    """
    # Handle ay ~ 0 (linear case)
    eps = 1e-12
    if abs(ay) < eps:
        if abs(vy0) < eps:
            return None
        t = -y0 / vy0
        return t if t > 0 else None
    # Quadratic
    a = 0.5 * ay
    b = vy0
    c = y0
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    r1 = (-b - math.sqrt(disc)) / (2*a)
    r2 = (-b + math.sqrt(disc)) / (2*a)
    roots = [t for t in (r1, r2) if t > 1e-10]
    if not roots:
        return None
    return min(roots)

def make_axes_equal(ax):
    ax.set_aspect('equal', adjustable='box')

def draw_arrow(ax, x0, y0, dx, dy, label, color='C0'):
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return
    # normalize for arrow style scaling
    arrow = FancyArrow(x0, y0, dx, dy, width=0.02*length, length_includes_head=True,
                       head_width=0.15*length, head_length=0.2*length, color=color, alpha=0.9)
    ax.add_patch(arrow)
    # label slightly beyond tip
    lx = x0 + dx * 1.05
    ly = y0 + dy * 1.05
    ax.text(lx, ly, label, fontsize=11, color=color, ha='left', va='center')

# -----------------------------
# Force Table UI
# -----------------------------
def default_forces_for_preset(mass: float) -> List[Force]:
    # Classroom example: 5 kg box, 20 N at 30° to horizontal, include weight + normal
    applied_fx, applied_fy = polar_to_cartesian(20.0, 30.0)
    forces = [
        Force("Applied", applied_fx, applied_fy, True),
        Force("Weight (mg)", 0.0, -mass*G, True),
        Force("Normal", 0.0, mass*G - applied_fy, True),  # simple static balance if pushing up
    ]
    return forces

def forces_data_editor(forces: List[Force]) -> List[Force]:
    st.markdown("#### Forces")
    st.caption("Add forces either by **components** or **magnitude+angle**. Use the quick‑add below or edit the table directly.")

    with st.expander("➕ Quick‑add force (polar input)", expanded=False):
        colp1, colp2, colp3, colp4 = st.columns([1.4, 1, 1, 1])
        name = colp1.text_input("Name", value="F1")
        mag = colp2.number_input("Magnitude F (N)", value=10.0, min_value=0.0, step=1.0)
        ang = colp3.number_input("Angle θ (° from +x)", value=0.0, step=5.0, format="%.1f")
        show = colp4.checkbox("Show on FBD", value=True)
        if st.button("Add force"):
            fx, fy = polar_to_cartesian(mag, ang)
            forces.append(Force(name, fx, fy, show))

    # Build editable table
    import pandas as pd
    df = pd.DataFrame([{"Name": f.name, "Fx (N)": f.fx, "Fy (N)": f.fy, "Show": f.show} for f in forces])
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn(width="medium"),
            "Fx (N)": st.column_config.NumberColumn(format="%.3f"),
            "Fy (N)": st.column_config.NumberColumn(format="%.3f"),
            "Show": st.column_config.CheckboxColumn(),
        }
    )
    # Convert back
    new_forces: List[Force] = []
    for _, row in edited.iterrows():
        try:
            new_forces.append(Force(str(row["Name"]), float(row["Fx (N)"]), float(row["Fy (N)"]), bool(row["Show"])))
        except Exception:
            pass
    return new_forces

# -----------------------------
# FBD Plot
# -----------------------------
def plot_fbd(forces: List[Force], scale_hint: float = 1.0):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    # Draw block
    block = Rectangle((-0.5, -0.25), 1.0, 0.5, linewidth=1.5, edgecolor='black', facecolor='#f2f2f2')
    ax.add_patch(block)
    ax.text(0, 0, "object", ha='center', va='center', fontsize=11)

    # Determine scaling for arrows
    maxF = max((math.hypot(f.fx, f.fy) for f in forces if f.show), default=1.0)
    if maxF < 1e-6:
        maxF = 1.0
    # Arrows originate at center (0,0)
    for i, f in enumerate(forces):
        if not f.show:
            continue
        scale = 0.9 / maxF  # keep arrows inside view
        draw_arrow(ax, 0, 0, f.fx*scale*scale_hint, f.fy*scale*scale_hint, f"{f.name}\n({f.fx:.1f}, {f.fy:.1f}) N", color=f"C{i%10}")

    # Style
    make_axes_equal(ax)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Free‑Body Diagram (vectors not to scale)")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Motion (analytic, constant accel)
# -----------------------------
def simulate_motion_1d(x0, vx0, ax, t_end, dt=0.01):
    t = np.arange(0.0, t_end + dt, dt)
    x = x0 + vx0*t + 0.5*ax*t*t
    vx = vx0 + ax*t
    return t, x, vx

def simulate_motion_2d(x0, y0, vx0, vy0, ax, ay, t_end, dt=0.01, stop_at_ground=True):
    # Determine t_end automatically if stop at ground and y0>=0 and ay<=0 etc.
    if stop_at_ground:
        t_hit = solve_time_to_y0(y0, vy0, ay)
        if t_hit is not None:
            t_end = min(t_end, t_hit)
    t = np.arange(0.0, t_end + dt, dt)
    x = x0 + vx0*t + 0.5*ax*t*t
    y = y0 + vy0*t + 0.5*ay*t*t
    vx = vx0 + ax*t
    vy = vy0 + ay*t
    return t, x, y, vx, vy

def plot_1d_graphs(t, x, vx, label_axis='x'):
    fig1, ax1 = plt.subplots(figsize=(6.0, 3.4))
    ax1.plot(t, x)
    ax1.set_xlabel("t (s)"); ax1.set_ylabel(f"{label_axis}(t) (m)")
    ax1.set_title(f"{label_axis}(t)")
    ax1.grid(True, alpha=0.25)
    st.pyplot(fig1); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6.0, 3.4))
    ax2.plot(t, vx)
    ax2.set_xlabel("t (s)"); ax2.set_ylabel(f"v{label_axis}(t) (m/s)")
    ax2.set_title(f"v{label_axis}(t)")
    ax2.grid(True, alpha=0.25)
    st.pyplot(fig2); plt.close(fig2)

def plot_2d_path(t, x, y):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(x, y)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Path")
    ax.grid(True, alpha=0.25)
    make_axes_equal(ax)
    st.pyplot(fig); plt.close(fig)

def run_animation_2d(x, y, fps=60, speed=1.0):
    """
    Frame‑rate independent animation in Streamlit using a placeholder.
    Physics arrays x,y are assumed dense. We display ~fps frames/sec
    but step through the arrays according to real elapsed time * speed.
    """
    ph = st.empty()
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    make_axes_equal(ax)

    # Precompute limits with margins
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xr = xmax - xmin if xmax > xmin else 1.0
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_xlim(xmin - 0.1*xr, xmax + 0.1*xr)
    ax.set_ylim(ymin - 0.1*yr, ymax + 0.1*yr)

    # Static trail
    path_line, = ax.plot([], [])
    point, = ax.plot([], [], marker='o')

    # Time mapping
    total_frames = int(len(x))
    T_sim = 1.0  # we'll map whole sim to 1 "sim-time" unit, then scale by speed
    t_index = 0
    start_real = time.perf_counter()
    last_frame_time = start_real

    while t_index < total_frames:
        now = time.perf_counter()
        elapsed = now - start_real
        # Map elapsed real time to an index
        sim_pos = min(1.0, elapsed * speed / (total_frames / fps / 1.0))  # linear mapping
        # Robust index step (monotonic)
        t_index = max(t_index, int(sim_pos * (total_frames - 1)))

        path_line.set_data(x[:t_index+1], y[:t_index+1])
        point.set_data([x[t_index]], [y[t_index]])

        ph.pyplot(fig)
        # Pace to target fps
        to_sleep = max(0.0, (1.0/fps) - (time.perf_counter() - last_frame_time))
        time.sleep(to_sleep)
        last_frame_time = time.perf_counter()

    plt.close(fig)

# -----------------------------
# Streamlit App
# -----------------------------
def app():
    st.title("Newton’s 2nd Law (1D/2D)")
    st.caption("Links forces to acceleration; bridges kinematics → dynamics.")

    # --------- Sidebar controls ---------
    st.sidebar.header("Setup")
    mode = st.sidebar.radio("Motion Mode", ["2D", "1D (x‑axis)"])
    ex = st.sidebar.toggle("Load classroom example", value=True)
    st.sidebar.divider()

    # Mass & initial conditions
    mass = st.sidebar.number_input("Mass (kg)", min_value=0.01, value=5.0, step=0.1, format="%.2f")
    x0 = st.sidebar.number_input("x₀ (m)", value=0.0, step=0.1)
    y0_default = 0.0 if mode != "2D" else 0.0
    y0 = 0.0 if mode != "2D" else st.sidebar.number_input("y₀ (m)", value=y0_default, step=0.1)
    vx0 = st.sidebar.number_input("vₓ₀ (m/s)", value=0.0, step=0.1)
    vy0 = 0.0 if mode != "2D" else st.sidebar.number_input("v_y₀ (m/s)", value=0.0, step=0.1)

    t_end = st.sidebar.number_input("Max time (s)", value=5.0, min_value=0.1, step=0.5)
    dt = st.sidebar.number_input("Physics step dt (s)", value=0.01, min_value=0.001, step=0.005, format="%.3f")
    stop_at_ground = st.sidebar.checkbox("Stop when y reaches 0", value=True) if mode == "2D" else False

    st.sidebar.divider()
    animate = st.sidebar.checkbox("Show animation", value=True)
    fps = st.sidebar.slider("Animation FPS", 10, 90, 60, step=10) if animate else 60
    speed = st.sidebar.slider("Playback speed (×)", 0.5, 5.0, 1.0, step=0.1) if animate else 1.0

    # --------- Forces ---------
    if ex:
        forces = default_forces_for_preset(mass)
    else:
        forces = [Force("Weight (mg)", 0.0, -mass*G, True)]

    forces = forces_data_editor(forces)

    # --------- Compute net force & acceleration ---------
    F_net_x = sum(f.fx for f in forces)
    F_net_y = sum(f.fy for f in forces)
    a_x = F_net_x / mass
    a_y = F_net_y / mass

    st.markdown("### Results")
    colA, colB, colC = st.columns(3)
    colA.metric("Net Force Fx (N)", f"{F_net_x:.3f}")
    colB.metric("Net Force Fy (N)", f"{F_net_y:.3f}")
    colC.metric("‖F_net‖ (N)", f"{math.hypot(F_net_x, F_net_y):.3f}")

    colA2, colB2, colC2 = st.columns(3)
    colA2.metric("aₓ (m/s²)", f"{a_x:.4f}")
    colB2.metric("a_y (m/s²)", f"{a_y:.4f}")
    colC2.metric("‖a‖ (m/s²)", f"{math.hypot(a_x, a_y):.4f}")

    # --------- FBD ---------
    st.markdown("### Free‑Body Diagram")
    plot_fbd(forces)

    # --------- Motion & Graphs ---------
    st.markdown("### Motion (constant acceleration)")
    if mode == "1D (x‑axis)":
        t, x, vx = simulate_motion_1d(x0, vx0, a_x, t_end, dt)
        plot_1d_graphs(t, x, vx, label_axis='x')
    else:
        t, x, y, vx, vy = simulate_motion_2d(x0, y0, vx0, vy0, a_x, a_y, t_end, dt, stop_at_ground=stop_at_ground)
        plot_2d_path(t, x, y)
        with st.expander("x(t), y(t), vₓ(t), v_y(t)"):
            # Compact 2x2 display without subplots constraint—render one by one
            plot_1d_graphs(t, x, vx, label_axis='x')
            plot_1d_graphs(t, y, vy, label_axis='y')

        if animate:
            st.markdown("#### Animation")
            st.info("Animation uses the precomputed physics arrays, so changing **dt** only affects accuracy, not playback smoothness.")
            run = st.button("▶️ Play / Replay")
            if run:
                run_animation_2d(x, y, fps=int(fps), speed=float(speed))

    # --------- Notes ---------
    with st.expander("Teacher notes / tips"):
        st.markdown(
            "- Use the quick‑add to enter a force by magnitude and angle.\n"
            "- The **Stop when y reaches 0** option ends motion at ground contact (useful for push‑up/push‑down cases).\n"
            "- The FBD arrows are **auto‑scaled** to fit near the block; numeric components appear on each label.\n"
            "- Animation playback is frame‑rate independent; physics uses **dt** you set, while display is paced by **FPS**."
        )

# For local debug
if __name__ == "__main__":
    app()
