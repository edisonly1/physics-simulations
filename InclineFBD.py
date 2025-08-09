import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import streamlit as st # type: ignore
import matplotlib.lines as mlines # type: ignore

def draw_incline_fbd(
    angle_deg=30,
    mass=2,
    mu=0,
    length=5,
    show_friction=False,
    direction="down",
    initial_velocity=0,
    show_applied=True,
):
    """
    direction: 'down' (default), or 'up' (or anything containing 'up' is treated as 'up')
    show_applied: If True, shows applied force for initial velocity > 0.
    """
    g = 10.0
    theta = np.radians(angle_deg)
    x0, y0 = 0, length * np.sin(theta)
    x1, y1 = length, 0

    frac = 1/3
    xb = x0 + frac * (x1 - x0)
    yb = y0 + frac * (y1 - y0)
    block_center = np.array([xb, yb])

    mg = mass * g
    fn = mg * np.cos(theta)
    f_friction = mu * fn if show_friction and mu > 0 else 0

    # --- Robust direction parsing ---
    direction_raw = str(direction).lower()
    if "up" in direction_raw:
        direction = "up"
    else:
        direction = "down"

    # --- Force direction vectors ---
    v_mg = np.array([0.0, -1.0])  # gravity: always down
    v_n = np.array([np.sin(theta), np.cos(theta)])  # normal: perp to incline

    ramp_vec = np.array([x1 - x0, y1 - y0])
    ramp_unit = ramp_vec / np.linalg.norm(ramp_vec)

    # Friction is always opposite motion
    if direction == "up":
        v_fric = ramp_unit  # friction points down ramp
    else:
        v_fric = -ramp_unit # friction points up ramp

    # Applied force: along ramp, direction of initial velocity
    show_applied_arrow = show_applied and abs(initial_velocity) > 0
    if show_applied_arrow:
        if direction == "up":
            v_applied = -ramp_unit  # up ramp (pushing up)
        else:
            v_applied = ramp_unit   # down ramp (pushing down)
        applied_magnitude = 0.7 * mg  # arbitrary visual length; not calculated from problem
    else:
        v_applied = np.array([0.0, 0.0])
        applied_magnitude = 0.0

    # --- Arrow scaling: max force gets target_length ---
    forces = [mg, fn]
    if show_friction and mu > 0:
        forces.append(f_friction)
    if show_applied_arrow:
        forces.append(applied_magnitude)
    max_force = max(forces) if forces else 1
    target_length = 1.0
    scale = target_length / max_force if max_force > 0 else 0

    arrow_mg = v_mg * mg * scale
    arrow_fn = v_n * fn * scale
    arrow_ffric = v_fric * f_friction * scale if (show_friction and mu > 0) else np.array([0.0, 0.0])
    arrow_applied = v_applied * applied_magnitude * scale if show_applied_arrow else np.array([0.0, 0.0])

    head_width = 0.08 * target_length
    head_length = 0.11 * target_length

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([x0, x1], [y0, y1], color="k", lw=4)
    ax.plot([x1 - 1, x1 + 1], [y1, y1], color="brown", lw=2)
    ax.plot(block_center[0], block_center[1], 'o', color="royalblue", markersize=12,
            markeredgecolor="black", markeredgewidth=2, label="Object", zorder=5)

    # --- Draw force arrows ---
    ax.arrow(*block_center, *arrow_mg, head_width=head_width, head_length=head_length,
             fc='green', ec='green', lw=3, length_includes_head=True)
    ax.arrow(*block_center, *arrow_fn, head_width=head_width, head_length=head_length,
             fc='orange', ec='orange', lw=3, length_includes_head=True)
    if show_friction and mu > 0:
        ax.arrow(*block_center, *arrow_ffric, head_width=head_width, head_length=head_length,
                 fc='brown', ec='brown', lw=3, length_includes_head=True, zorder=4)
    if show_applied_arrow:
        ax.arrow(*block_center, *arrow_applied, head_width=head_width, head_length=head_length,
                 fc='blue', ec='blue', lw=3, length_includes_head=True, zorder=3)

    # --- Legend ---
    legend_handles = [
        mlines.Line2D([], [], color='green', lw=3, label='mg (gravity)'),
        mlines.Line2D([], [], color='orange', lw=3, label='N (normal)'),
    ]
    if show_friction and mu > 0:
        legend_handles.append(mlines.Line2D([], [], color='brown', lw=3, label=r'$f_k$ (friction)'))
    if show_applied_arrow:
        legend_handles.append(mlines.Line2D([], [], color='blue', lw=3, label='applied (push)'))

    ax.legend(handles=legend_handles, loc='upper left', fontsize=13, frameon=True)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Free Body Diagram", fontsize=18, weight='bold')
    margin = 1.0
    ax.set_xlim(block_center[0] - margin, block_center[0] + margin)
    ax.set_ylim(block_center[1] - margin, block_center[1] + margin)
    plt.tight_layout()
    st.pyplot(fig)
