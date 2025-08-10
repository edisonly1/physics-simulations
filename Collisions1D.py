# Collisions1D.py
# Streamlit module: 1D Elastic & (Perfectly) Inelastic Collisions
# Features: sliders for m1, m2, u1, u2; elastic/inelastic toggle; time scrubber animation;
# outputs final velocities + momentum/KE checks; energy pie charts before & after.

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Physics helpers (pure functions -> easy to unit test)
# -----------------------------
def final_velocities(m1: float, m2: float, u1: float, u2: float, mode: str) -> tuple[float, float]:
    """Return v1, v2 after collision for 'Elastic' or 'Perfectly Inelastic'."""
    if mode == "Elastic":
        v1 = ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)
        v2 = ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)
        return float(v1), float(v2)
    elif mode == "Perfectly Inelastic":
        v = (m1 * u1 + m2 * u2) / (m1 + m2)
        return float(v), float(v)
    else:
        raise ValueError("mode must be 'Elastic' or 'Perfectly Inelastic'")

def will_collide(x10: float, x20: float, u1: float, u2: float) -> bool:
    """1D point-mass check: do worldlines intersect for t>0?"""
    # Need u1 > u2 and x10 < x20 to collide (cart 1 behind cart 2).
    return (x10 < x20) and (u1 > u2)

def collision_time(x10: float, x20: float, u1: float, u2: float) -> float | None:
    """Time when x1(t) == x2(t); None if no collision for t>0."""
    if u1 == u2:
        return None
    t = (x20 - x10) / (u1 - u2)
    return t if t > 0 else None

def positions_over_time(t: np.ndarray, x10: float, x20: float,
                        u1: float, u2: float, m1: float, m2: float,
                        mode: str) -> tuple[np.ndarray, np.ndarray, float | None, float, float]:
    """Piecewise motion pre/post collision; returns x1(t), x2(t), t_c, v1f, v2f."""
    t_c = collision_time(x10, x20, u1, u2)
    v1f, v2f = final_velocities(m1, m2, u1, u2, mode)

    x1 = np.empty_like(t)
    x2 = np.empty_like(t)

    if t_c is None:
        # No collision in future: uniform motion for both
        x1 = x10 + u1 * t
        x2 = x20 + u2 * t
        return x1, x2, None, v1f, v2f

    # Pre-collision segments
    pre = t <= t_c
    post = t > t_c
    x1[pre] = x10 + u1 * t[pre]
    x2[pre] = x20 + u2 * t[pre]

    # Collision position at t_c using pre-motion
    x_c = x10 + u1 * t_c  # = x20 + u2 * t_c

    # Post-collision segments
    dt = t[post] - t_c
    if mode == "Perfectly Inelastic":
        # Both share same position after t_c
        x_after = x_c + v1f * dt  # v1f == v2f
        x1[post] = x_after
        x2[post] = x_after
    else:
        x1[post] = x_c + v1f * dt
        x2[post] = x_c + v2f * dt

    return x1, x2, t_c, v1f, v2f

def energies_and_momenta(m1, m2, u1, u2, v1, v2):
    KEi = 0.5 * m1 * u1**2 + 0.5 * m2 * u2**2
    KEf = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    pi = m1 * u1 + m2 * u2
    pf = m1 * v1 + m2 * v2
    return KEi, KEf, pi, pf

# -----------------------------
# Plot helpers
# -----------------------------
def draw_track_and_carts(x1, x2, L=12.0, cart_w=0.9, cart_h=0.45) -> plt.Figure:
    """Return a Matplotlib figure showing two carts on a horizontal track."""
    fig, ax = plt.subplots(figsize=(8, 2.6))
    # Track
    ax.plot([0, L], [0, 0], linewidth=6, alpha=0.25)
    # Carts as rectangles centered at x positions
    for x, color in zip([x1, x2], ["tab:blue", "tab:orange"]):
        left = x - cart_w / 2
        rect = plt.Rectangle((left, 0.05), cart_w, cart_h, ec="black", fc=color, alpha=0.9)
        ax.add_patch(rect)
        ax.plot([x], [0.05 + cart_h + 0.02], marker="v", ms=6)  # little marker above each cart
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Position (m)")
    ax.get_yaxis().set_visible(False)
    ax.set_title("1D Collision Animation (use the time slider)")
    fig.tight_layout()
    return fig

def pie_two_panels(ke1_i, ke2_i, ke1_f, ke2_f) -> plt.Figure:
    """Two pie charts side-by-side: energy distribution before & after."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
    total_i = ke1_i + ke2_i
    total_f = ke1_f + ke2_f

    # Avoid zero-sum pies by adding tiny epsilon when needed
    eps = 1e-9
    axs[0].pie([max(ke1_i, eps), max(ke2_i, eps)],
               labels=[f"Cart 1\n{ke1_i:.2f} J", f"Cart 2\n{ke2_i:.2f} J"],
               autopct=lambda p: f"{p:.0f}%" if total_i > 0 else "")
    axs[0].set_title("Energy BEFORE")

    axs[1].pie([max(ke1_f, eps), max(ke2_f, eps)],
               labels=[f"Cart 1\n{ke1_f:.2f} J", f"Cart 2\n{ke2_f:.2f} J"],
               autopct=lambda p: f"{p:.0f}%" if total_f > 0 else "")
    axs[1].set_title("Energy AFTER")
    fig.tight_layout()
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
def app():
    st.title("Elastic & Inelastic Collisions (1D)")

    with st.expander("Show core equations"):
        st.latex(r"p\ \text{conserved: } m_1 u_1 + m_2 u_2 = m_1 v_1 + m_2 v_2")
        st.latex(r"\text{Elastic only: } K E_i = K E_f")
        st.latex(r"v_1^{(\mathrm{elastic})}=\frac{(m_1-m_2)u_1+2m_2u_2}{m_1+m_2},\quad"
                 r"v_2^{(\mathrm{elastic})}=\frac{(m_2-m_1)u_2+2m_1u_1}{m_1+m_2}")
        st.latex(r"v^{(\mathrm{perfect\ inelastic})}=\frac{m_1 u_1+m_2 u_2}{m_1+m_2}")

    # --- Inputs
    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader("Inputs")
        m1 = st.slider("Mass m₁ (kg)", 0.1, 10.0, 1.0, 0.1)
        m2 = st.slider("Mass m₂ (kg)", 0.1, 10.0, 1.0, 0.1)
        u1 = st.slider("Initial velocity u₁ (m/s)", -10.0, 10.0, 4.0, 0.1)
        u2 = st.slider("Initial velocity u₂ (m/s)", -10.0, 10.0, 0.0, 0.1)
        mode = st.radio("Collision type", ["Elastic", "Perfectly Inelastic"], horizontal=True)

        # Geometry/time controls
        st.markdown("**Geometry & Time**")
        L = st.slider("Track length (m)", 6.0, 20.0, 12.0, 0.5)
        separation = st.slider("Initial separation (m) — center₂ minus center₁", 1.0, L - 1.0, min(6.0, L - 2.0), 0.1)
        duration = st.slider("Duration (s)", 1.0, 12.0, 6.0, 0.1)

    # Derived starting positions
    x10 = 0.8  # start near the left
    x20 = min(x10 + separation, L - 0.8)

    # Time base + scrubber
    t_end = duration
    t = np.linspace(0.0, t_end, 400)
    t_now = st.slider("Time (s)", 0.0, float(t_end), 0.0, 0.01)

    # Compute motion
    x1_path, x2_path, t_c, v1f, v2f = positions_over_time(t, x10, x20, u1, u2, m1, m2, mode)

    # Draw animation frame at t_now
    # Choose nearest precomputed sample (good enough for display; physics still exact in arrays)
    idx = int(np.clip(np.searchsorted(t, t_now), 0, len(t) - 1))
    fig_anim = draw_track_and_carts(x1_path[idx], x2_path[idx], L=L)
    st.pyplot(fig_anim, use_container_width=True)

    # --- Outputs
    with colR:
        st.subheader("Results")
        KEi, KEf, pi, pf = energies_and_momenta(m1, m2, u1, u2, v1f, v2f)

        # Final velocities
        st.markdown(f"**Final velocities**  \n"
                    f"v₁ = `{v1f:.3f}` m/s,   v₂ = `{v2f:.3f}` m/s")

        # Momentum/energy checks
        st.markdown(f"**Momentum**: pᵢ = `{pi:.3f}` kg·m/s,  p_f = `{pf:.3f}` kg·m/s  "
                    f"(Δp = `{pf - pi:+.3e}`)")
        st.markdown(f"**Kinetic Energy**: KEᵢ = `{KEi:.3f}` J,  KE_f = `{KEf:.3f}` J  "
                    f"(ΔKE = `{KEf - KEi:+.3f}` J,  {100*(KEf-KEi)/KEi:+.1f}% change"
                    f"{'' if KEi>1e-12 else ' — initial KE≈0'} )")

        # Collision timing info
        if will_collide(x10, x20, u1, u2):
            if t_c is not None and 0.0 <= t_c <= t_end:
                st.info(f"Collision occurs at **t = {t_c:.3f} s** at x ≈ {x10 + u1*t_c:.2f} m.")
            else:
                st.warning("They would collide, but **not within** the selected duration.")
        else:
            st.warning("With these inputs, the carts **do not collide** (worldlines never meet for t>0).")

    # Energy pies (before/after + per-cart breakdown)
    ke1_i, ke2_i = 0.5 * m1 * u1**2, 0.5 * m2 * u2**2
    ke1_f, ke2_f = 0.5 * m1 * v1f**2, 0.5 * m2 * v2f**2
    fig_pies = pie_two_panels(ke1_i, ke2_i, ke1_f, ke2_f)
    st.pyplot(fig_pies, use_container_width=True)

    # Footnotes for teachers
    st.caption(
        "Notes: 1) Motion is piecewise-uniform with an instantaneous collision at t_c. "
        "2) We model 1D point masses; rectangles are for visualization only. "
        "3) Perfectly inelastic = carts stick (shared velocity)."
    )

# Allow running as a standalone page for quick testing
if __name__ == "__main__":
    st.set_page_config(page_title="1D Collisions", layout="wide")
    app()
