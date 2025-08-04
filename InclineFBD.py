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

    # Forces (using fixed lengths for clarity)
    mg_len = 0.8
    fn_len = 0.8
    fp_len = 0.8
    ffric_len = 0.7

    mg = mass * g
    fn = mg * np.cos(theta)
    fp = mg * np.sin(theta)
    f_friction = mu * fn if show_friction and mu > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([x0, x1], [y0, y1], color="k", lw=4)
    ax.plot([x1 - 1, x1 + 1], [y1, y1], color="brown", lw=2)
    ax.plot(block_rot[:, 0], block_rot[:, 1], color="royalblue", lw=2)
    ax.fill(block_rot[:, 0], block_rot[:, 1], "royalblue", alpha=0.7)

    block_center = np.array([xb, yb])

    # Gravity (down)
    mg_tip = (block_center[0], block_center[1] - mg_len)
    ax.arrow(*block_center, 0, -mg_len, head_width=0.13, head_length=0.18, fc='green', ec='green', lw=3, length_includes_head=True)
    ax.text(mg_tip[0], mg_tip[1] - 0.12, r"$mg$", color="green", fontsize=18, ha="center", va="top")

    # Normal (perpendicular)
    nx = np.sin(theta)
    ny = np.cos(theta)
    fn_tip = (block_center[0] + nx*fn_len, block_center[1] + ny*fn_len)
    ax.arrow(*block_center, nx*fn_len, ny*fn_len, head_width=0.13, head_length=0.18, fc='orange', ec='orange', lw=3, length_includes_head=True)
    ax.text(fn_tip[0] + 0.04, fn_tip[1] + 0.04, r"$N$", color="orange", fontsize=18, ha="left", va="bottom")

    # Parallel (down ramp)
    px = np.cos(theta)
    py = -np.sin(theta)
    fp_tip = (block_center[0] + px*fp_len, block_center[1] + py*fp_len)
    ax.arrow(*block_center, px*fp_len, py*fp_len, head_width=0.13, head_length=0.18, fc='red', ec='red', lw=3, length_includes_head=True)
    ax.text(fp_tip[0] + 0.04, fp_tip[1] - 0.04, r"$F_{\parallel}$", color="red", fontsize=18, ha="left", va="top")

    # Friction (if present, up the ramp)
    if show_friction and mu > 0:
        ffric_tip = (block_center[0] - px*ffric_len, block_center[1] - py*ffric_len)
        ax.arrow(*block_center, -px*ffric_len, -py*ffric_len, head_width=0.13, head_length=0.18, fc='brown', ec='brown', lw=3, length_includes_head=True)
        ax.text(ffric_tip[0] - 0.04, ffric_tip[1] + 0.04, r"$f_k$", color="brown", fontsize=18, ha="right", va="bottom")

    ax.set_aspect("equal")
    ax.axis("off")
    margin = 1.2
    ax.set_xlim(block_center[0] - margin, block_center[0] + margin)
    ax.set_ylim(block_center[1] - margin, block_center[1] + margin)
    ax.set_title("Free-Body Diagram (FBD) for Block on Incline", fontsize=18, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)
