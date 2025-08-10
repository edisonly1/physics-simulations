# WorkEnergy.py
# Streamlit module: Work–Energy Theorem (AP Physics 1)
# Features:
# - Constant force: W = F d cos(theta)
# - Variable force: W = ∫ F(x) dx (spring or custom expression)
# - F–x graph with shaded work area
# - Outputs: W_net, ΔKE, final speed
# - Optional CSV export of x, F(x), cumulative work

from __future__ import annotations
import math
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# ---------- Math helpers ----------

def trapz_integral(func, a: float, b: float, n: int = 1000) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (W, x, F, W_cum): integral of func from a to b via trapezoidal rule."""
    if n < 2:
        n = 2
    x = np.linspace(a, b, n)
    F = func(x)
    # cumulative work vs x (same orientation as x)
    W_cum = np.cumtrapz(F, x, initial=0.0)
    W = float(np.trapz(F, x))
    return W, x, F, W_cum

def final_speed_from_work(m: float, v0: float, W: float) -> tuple[float, bool]:
    """Compute final speed from ΔKE = W. Returns (vf, stopped_early_flag)."""
    E0 = 0.5 * m * v0 * v0
    E1 = E0 + W
    if E1 <= 0:
        return 0.0, True
    return float(math.sqrt(2.0 * E1 / m)), False

# Safe eval environment for custom F(x)
_ALLOWED = {
    "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "sqrt": np.sqrt,
    "pi": np.pi, "abs": np.abs, "sign": np.sign, "where": np.where, "log": np.log
}

def make_force_from_expr(expr: str):
    """Return a vectorized force function f(x) from a text expression in x."""
    # Very restricted eval: no builtins, only numpy+functions above, and 'x' variable
    def f(x):
        local = _ALLOWED.copy()
        local["x"] = x
        return eval(expr, {"__builtins__": {}}, local)
    # quick test to fail fast with helpful error
    try:
        _ = f(np.array([0.0, 1.0]))
    except Exception as e:
        raise ValueError(f"Invalid expression for F(x): {e}")
    return f

# ---------- Plot helpers ----------

def plot_force_area(x: np.ndarray, F: np.ndarray, x0: float, x1: float, W: float, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, F, linewidth=2, label="Force $F(x)$")
    xmin, xmax = min(x0, x1), max(x0, x1)

    # Shade positive/negative portions distinctly
    ax.fill_between(x, 0, F, where=(x >= xmin) & (x <= xmax) & (F >= 0), alpha=0.35, step="mid", label="Positive work")
    ax.fill_between(x, 0, F, where=(x >= xmin) & (x <= xmax) & (F < 0),  alpha=0.25, step="mid", label="Negative work")

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Displacement $x$ (m)")
    ax.set_ylabel("Force $F$ (N)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate net area
    ax.text(0.02, 0.04, f"Shaded area = Work = {W:.3g} J", transform=ax.transAxes)
    fig.tight_layout()
    return fig

def plot_cumulative_work(x: np.ndarray, W_cum: np.ndarray, x0: float, x1: float, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(x, W_cum, linewidth=2)
    ax.axvline(x0, linestyle="--", linewidth=1)
    ax.axvline(x1, linestyle="--", linewidth=1)
    ax.set_xlabel("Displacement $x$ (m)")
    ax.set_ylabel("Cumulative work $W(x)$ (J)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ---------- UI / App ----------

def app():
    st.title("Work–Energy Theorem")
    st.caption("Relate net work to change in kinetic energy.  $W_{\\text{net}} = \\Delta KE$")

    with st.expander("Show formulas"):
        st.latex(r"W_{\text{net}} = \Delta KE = \tfrac12 m v_f^2 - \tfrac12 m v_0^2")
        st.latex(r"W = F\,d\cos\theta \quad \text{(constant force along a straight path)}")
        st.latex(r"W = \int_{x_0}^{x_1} F(x)\,dx \quad \text{(variable force)}")

    # Common inputs
    left, right = st.columns([1, 1])
    with left:
        m = st.slider("Mass m (kg)", 0.1, 20.0, 2.0, 0.1)
        v0 = st.slider("Initial speed v₀ (m/s)", 0.0, 50.0, 0.0, 0.1)

    mode = st.radio("Force model", ["Constant force", "Variable force (F(x))"], horizontal=True)

    if mode == "Constant force":
        with right:
            Fmag = st.slider("Force magnitude F (N)", 0.0, 300.0, 50.0, 0.5)
            theta = st.slider("Angle θ between F and displacement (deg)", 0.0, 180.0, 0.0, 1.0)
            d = st.slider("Displacement d (m)", -20.0, 20.0, 5.0, 0.1)

        # Work and final speed
        W = float(Fmag * d * math.cos(math.radians(theta)))
        vf, stopped = final_speed_from_work(m, v0, W)

        # Build a simple F-x profile for graphing (constant)
        x0 = 0.0
        x1 = d
        xs = np.linspace(min(x0, x1), max(x0, x1), 400)
        Fline = np.full_like(xs, Fmag * math.cos(math.radians(theta)) * np.sign(d) if d != 0 else 0.0)

        fig_fx = plot_force_area(xs, Fline, x0, x1, W, "Force vs Displacement (constant)")
        st.pyplot(fig_fx, use_container_width=True)

        # Outputs
        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Net work W", f"{W:.3g} J")
        c2.metric("ΔKE", f"{W:.3g} J")
        c3.metric("Final speed v_f", f"{vf:.3g} m/s")
        if stopped:
            st.warning("Initial kinetic energy + work ≤ 0 → the object would stop before completing the motion.")

    else:
        # Variable force UI
        with right:
            model = st.selectbox("Force function", ["Spring: F = -k (x - x_eq)", "Custom expression in x"])
            x0 = st.slider("Start position x₀ (m)", -10.0, 10.0, -0.5, 0.1)
            dx = st.slider("Displacement Δx (m)", -10.0, 10.0, 1.0, 0.1)
            x1 = x0 + dx

        if model.startswith("Spring"):
            k = st.slider("Spring constant k (N/m)", 1.0, 500.0, 100.0, 1.0)
            xeq = st.slider("Equilibrium position x_eq (m)", -5.0, 5.0, 0.0, 0.1)

            def F_spring(x):
                return -k * (x - xeq)

            # Integration window (pad a bit around the motion interval)
            xmin, xmax = min(x0, x1), max(x0, x1)
            pad = max(0.5, 0.1 * (xmax - xmin if xmax != xmin else 1.0))
            W, xs, Fs, Wcum = trapz_integral(F_spring, xmin - pad, xmax + pad, n=1200)

            # But we only want work across x0->x1:
            W_motion, _, _, _ = trapz_integral(F_spring, x0, x1, n=1200)
            fig_fx = plot_force_area(xs, Fs, x0, x1, W_motion, "Force vs Displacement (spring)")
            st.pyplot(fig_fx, use_container_width=True)

            W_use = W_motion

        else:
            st.markdown(
                "Enter **F(x)** using `x` and numpy functions, e.g. `100 - 20*x`, `50*sin(x)`, or `-k*(x-0.2)` "
                "(if you include symbols like `k`, give them numeric values directly)."
            )
            expr = st.text_input("F(x) =", value="100 - 20*x")
            try:
                f_custom = make_force_from_expr(expr)
                xmin, xmax = min(x0, x1), max(x0, x1)
                pad = max(0.5, 0.1 * (xmax - xmin if xmax != xmin else 1.0))
                # Prepare a plotting window around motion interval
                xs = np.linspace(xmin - pad, xmax + pad, 1000)
                Fs = f_custom(xs)
                # Work only over the motion interval:
                W_use, xm, Fm, Wcum_m = trapz_integral(f_custom, x0, x1, n=1200)

                fig_fx = plot_force_area(xs, Fs, x0, x1, W_use, "Force vs Displacement (custom)")
                st.pyplot(fig_fx, use_container_width=True)
            except Exception as e:
                st.error(str(e))
                return

        # Common results for variable-force case
        vf, stopped = final_speed_from_work(m, v0, W_use)
        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Net work W", f"{W_use:.3g} J")
        c2.metric("ΔKE", f"{W_use:.3g} J")
        c3.metric("Final speed v_f", f"{vf:.3g} m/s")
        if stopped:
            st.warning("Initial kinetic energy + work ≤ 0 → the object would stop before completing the motion.")

        # Optional cumulative work plot & CSV
        with st.expander("Show cumulative work plot and download data"):
            # Build a dense profile across the motion interval for cumulative work and CSV
            f_for_csv = F_spring if model.startswith("Spring") else f_custom
            x_dense = np.linspace(min(x0, x1), max(x0, x1), 1200)
            F_dense = f_for_csv(x_dense)
            Wcum_dense = np.cumtrapz(F_dense, x_dense, initial=0.0)
            fig_wc = plot_cumulative_work(x_dense, Wcum_dense, x0, x1, "Cumulative work across the motion")
            st.pyplot(fig_wc, use_container_width=True)

            df = pd.DataFrame({"x": x_dense, "F(x)": F_dense, "W_cumulative(x)": Wcum_dense})
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (x, F, W_cumulative)", data=csv_bytes, file_name="work_energy_profile.csv", mime="text/csv")

    st.info("Tip: Positive shaded area adds energy; negative shaded area removes energy.  "
            "For springs, area equals the change in elastic potential energy, \( \Delta U_s = \tfrac12 k(x_1 - x_{eq})^2 - \tfrac12 k(x_0 - x_{eq})^2 \).")

# Allow running directly (handy for local testing)
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    stcli.main(["run", __file__])
