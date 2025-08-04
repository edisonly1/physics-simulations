import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def draw_incline_fbd(angle_deg=30, mass=2, mu=0, length=5, show_friction=False):
    g = 10.0
    theta = np.radians(angle_deg)
    # Ramp: top-left to bottom-right
    x0, y0 = 0, length * np.sin(theta)
    x1, y1 = length, 0

    # Block position (1/3 along ramp)
    frac = 1/3
    xb = x0 + frac * (x1 - x0)
    yb = y0 + frac * (y1 - y0)
    block_size = 0.3

    # Block (rotated square)
    block_angle = -theta
    block = np.array([
        [-block_size/2, -block_size/2],
        [ block_size/2, -block_size/2],
        [ block_size/2,  block_size/2],
        [-block_size/2,  block_size/2],
        [-block_size/2, -block_size/2]
    ])
    rot = np.array([
        [np.cos(block_angle), -np.sin(block_angle)],
        [np.sin(block_angle),  np.cos(block_angle)]
    ])
    block_rot = block @ rot.T + [xb, yb]

    # Compute actual force magnitudes
    mg = mass * g  # gravity magnitude
    fn = mg * np.cos(theta)  # normal
    fp = mg * np.sin(theta)  # parallel (down ramp)
    f_friction = mu * fn if show_friction and mu > 0 else 0  # friction up ramp

    # Define direction unit vectors
    # Gravity: straight down
    v_mg = np.array([0.0, -1.0])
    # Normal: perpendicular out of ramp
    v_n = np.array([np.sin(theta), np.cos(theta)])
    # Parallel: down the ramp
    v_p = np.array([np.cos(theta), -np.sin(theta)])
    # Friction: opposite parallel
    v_fric = -v_p

    # Determine scaling so largest force maps to a reasonable arrow length
    forces = [mg, fn, fp]
    if show_friction and mu > 0:
        forces.append(f_friction)
    max_force = max(forces)
    target_length = 1.0  # length in plot units for the largest force
    scale = target_length / max_force if max_force > 0 else 0

    # Scaled vectors
    arrow_mg = v_mg * mg * scale
    arrow_fn = v_n * fn * scale
    arrow_fp = v_p * fp * scale
    arrow_ffric = v_fric * f_friction * scale if (show_friction and mu > 0) else np.array([0.0, 0.0])

    # Arrow head sizing (relative to scale)
    head_width = 0.08 * target_length  # constant small
    head_length = 0.11 * target_length

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([x0, x1], [y0, y1], color="k", lw=4)
    ax.plot([x1 - 1, x1 + 1], [y1, y1], color="brown", lw=2)
    ax.plot(block_rot[:, 0], block_rot[:, 1], color="royalblue", lw=2)
    ax.fill(block_rot[:, 0], block_rot[:, 1], "royalblue", alpha=0.7)

    block_center = np.array([xb, yb])

    # Gravity
    ax.arrow(*block_center, *arrow_mg, head_width=head_width, head_length=head_length,
             fc='green', ec='green', lw=3, length_includes_head=True)
    mg_tip = block_center + arrow_mg
    ax.text(mg_tip[0], mg_tip[1] - 0.05, r"$mg$", color="green", fontsize=16, ha="center", va="top")

    # Normal
    ax.arrow(*block_center, *arrow_fn, head_width=head_width, head_length=head_length,
             fc='orange', ec='orange', lw=3, length_includes_head=True)
    fn_tip = block_center + arrow_fn
    ax.text(fn_tip[0] + 0.05, fn_tip[1] + 0.05, r"$N$", color="orange", fontsize=16, ha="left", va="bottom")

    # Parallel
    ax.arrow(*block_center, *arrow_fp, head_width=head_width, head_length=head_length,
             fc='red', ec='red', lw=3, length_includes_head=True)
    fp_tip = block_center + arrow_fp
    ax.text(fp_tip[0] + 0.04, fp_tip[1] - 0.04, r"$F_{\parallel}$", color="red", fontsize=16, ha="left", va="top")

    # Friction, if any
    if show_friction and mu > 0:
        ax.arrow(*block_center, *arrow_ffric, head_width=head_width, head_length=head_length,
                 fc='brown', ec='brown', lw=3, length_includes_head=True)
        ffric_tip = block_center + arrow_ffric
        ax.text(ffric_tip[0] - 0.04, ffric_tip[1] + 0.04, r"$f_k$", color="brown", fontsize=16, ha="right", va="bottom")

    # Optionally: annotate numerical magnitudes near each vector
    # e.g., show scaled numeric values (uncomment if desired)
    # ax.text(mg_tip[0], mg_tip[1] - 0.2, f"{mg:.2f} N", color="green", fontsize=12, ha="center")
    # ax.text(fn_tip[0] + 0.1, fn_tip[1] + 0.1, f"{fn:.2f} N", color="orange", fontsize=12)
    # ax.text(fp_tip[0] + 0.1, fp_tip[1] - 0.1, f"{fp:.2f} N", color="red", fontsize=12)
    # if show_friction and mu > 0:
    #     ax.text(ffric_tip[0] - 0.1, ffric_tip[1] + 0.1, f"{f_friction:.2f} N", color="brown", fontsize=12)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Free Body Diagram", fontsize=15, weight='bold') 
    plt.tight_layout() 
    st.pyplot(fig)
    margin = 1.0
    ax.set_xlim(block_center[0] - margin, block_center[0] + margin)
    ax.set_ylim(block_center[1] - margin, block_center[1] + margin)