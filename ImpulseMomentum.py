# ImpulseMomentum.py
# Streamlit module: Impulse–Momentum Theorem (AP Physics 1)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

TOL = 1e-9

# ---------------- math helpers ----------------
def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum((y[1:] + y[:-1]) * 0.5 * np.diff(x))
    return out

def show_fig(fig):
    # Render then close so Streamlit doesn't keep a handle to a stale PNG id
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    try:
        plt.close(fig)
    except Exception:
        pass


def build_preset_force(shape: str, Fmax: float, duration: float, pre: float, post: float, dt: float):
    T = max(1e-3, pre + duration + post)
    N = max(200, int(round(T/dt)))
    t = np.linspace(0.0, T, N)
    F = np.zeros_like(t)
    t0, t1 = pre, pre + duration
    if shape == "Rectangular":
        F[(t >= t0) & (t <= t1)] = Fmax
    elif shape == "Triangular":
        mid = (t0 + t1) / 2.0
        up = (t >= t0) & (t <= mid)
        dn = (t >= mid) & (t <= t1)
        if np.any(up): F[up] = Fmax * (t[up] - t0) / max(TOL, (mid - t0))
        if np.any(dn): F[dn] = Fmax * (t1 - t[dn]) / max(TOL, (t1 - mid))
    elif shape == "Half-sine":
        mask = (t >= t0) & (t <= t1)
        tau = (t[mask] - t0) / max(TOL, duration)
        F[mask] = Fmax * np.sin(np.pi * tau)
    else:
        F[(t >= t0) & (t <= t1)] = Fmax
    return t, F

def build_custom_force(points_df: pd.DataFrame, total_time: float, dt: float):
    df = points_df.copy()
    df["t"] = df["t"].clip(lower=0.0, upper=max(1e-3, total_time))
    df = df.sort_values("t", kind="mergesort").drop_duplicates(subset="t")
    if df["t"].iloc[0] > 0.0: df = pd.concat([pd.DataFrame([{"t": 0.0, "F": 0.0}]), df], ignore_index=True)
    if df["t"].iloc[-1] < total_time: df = pd.concat([df, pd.DataFrame([{"t": total_time, "F": 0.0}])], ignore_index=True)
    t = np.linspace(0.0, total_time, max(200, int(round(total_time/dt))))
    F = np.interp(t, df["t"].to_numpy(), df["F"].to_numpy())
    return t, F, df

def impulse_and_kinematics(t: np.ndarray, F: np.ndarray, m: float, v0: float):
    J_total = float(np.trapz(F, t))
    mask = np.abs(F) > TOL
    if np.count_nonzero(mask) >= 2:
        t_contact = t[mask]; F_contact = F[mask]
        J_contact = float(np.trapz(F_contact, t_contact))
        dt_contact = float(t_contact[-1] - t_contact[0])
        F_avg = J_contact / dt_contact if dt_contact > 0 else 0.0
    else:
        J_contact, F_avg = 0.0, 0.0
    a = F / max(TOL, m)
    dv = cumtrapz(a, t)
    v = v0 + dv
    x = 0.0 + cumtrapz(v, t)
    p = m * v
    return {"J_total": J_total, "J_contact": J_contact, "F_avg": F_avg, "v": v, "x": x, "p": p, "a": a}

# ---------------- plotting helpers ----------------
def _idx_at_time(t: np.ndarray, t_now: float) -> int:
    return int(np.clip(np.searchsorted(t, t_now, side="right") - 1, 0, len(t) - 1))

def plot_force_until(t, F, t_now):
    i = _idx_at_time(t, t_now)
    J_now = float(np.trapz(F[:i+1], t[:i+1])) if i >= 1 else 0.0
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(t, F, lw=2, label="Force $F(t)$")
    if i >= 1:
        ax.fill_between(t[:i+1], 0, F[:i+1], alpha=0.30, label=f"Impulse so far  J(t) = {J_now:.3f} N·s")
    ax.axvline(t_now, color="k", lw=1, alpha=0.7)
    ax.axhline(0, lw=1, color="#888")
    ax.set_xlabel("time (s)"); ax.set_ylabel("force (N)")
    leg = ax.legend(loc="best")          # <-- safer than passing title kwarg
    # Optional legend title:
    # leg.set_title("Impulse view")
    ax.grid(alpha=0.25); fig.tight_layout()
    return fig, J_now

def plot_velocity_until(t, v, v0, t_now, J_total, m):
    i = _idx_at_time(t, t_now)
    v_now = float(v[i])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(t, v, lw=2, label="Velocity $v(t)$")
    ax.axhline(v0, ls="--", lw=1, label=f"$v_0$ = {v0:.3f} m/s")
    ax.axvline(t_now, color="k", lw=1, alpha=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("velocity (m/s)")
    leg = ax.legend(loc="best")
    # Show Δv info as a small textbox instead of legend title (compat-safe)
    ax.text(0.02, 0.98, f"Δv = J/m = {J_total/m:.3f} m/s", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", alpha=0.15))
    ax.grid(alpha=0.25); fig.tight_layout()
    return fig

def draw_ball_panel_adv(t, x, v, F, t_now):
    i = _idx_at_time(t, t_now)
    x_now, v_now, F_now = float(x[i]), float(v[i]), float(F[i])
    fig, ax = plt.subplots(figsize=(7, 2.8))
    x_span = max(1.0, np.max(x) - np.min(x)); pad = 0.12 * x_span
    ax.set_xlim(np.min(x) - pad, np.max(x) + pad); ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([]); ax.set_xlabel("position (m) — trail ∝ |v|, pulse ∝ |F|")
    ax.grid(alpha=0.12, axis="x")
    trail_len = 0.18 * (np.abs(v_now) / (np.max(np.abs(v)) + TOL)) * (np.max(x) - np.min(x) + TOL)
    ax.plot([x_now - trail_len, x_now], [0, 0], lw=6, alpha=0.35)
    pulse_h = 0.9 * (np.abs(F_now) / (np.max(np.abs(F)) + TOL))
    ax.fill_between([x_now - 0.02*x_span, x_now + 0.02*x_span], -pulse_h, 0, alpha=0.35)
    ax.scatter([x_now], [0], s=420, edgecolor="k", linewidths=1.2)
    fig.tight_layout()
    return fig

# ---------------- Streamlit UI ----------------
def app():
    st.title("Impulse–Momentum Theorem")
    st.caption("Link between force over time and momentum change:  "
               "$J = \\Delta p = F_{avg}\\,\\Delta t = \\int F(t)\\,dt$.")

    colL, colR = st.columns([1, 2])

    with colL:
        st.subheader("Inputs")
        m = st.number_input("Mass m (kg)", min_value=0.01, value=0.20, step=0.01, format="%.2f")
        v0 = st.number_input("Initial velocity v₀ (m/s)", value=0.0, step=0.1, format="%.2f")
        mode = st.radio("Force profile", ["Preset pulse", "Custom (table)"], index=0)
        dt = st.number_input("Time step dt (s)", min_value=0.0005, value=0.002, step=0.0005, format="%.4f")

        if mode == "Preset pulse":
            shape = st.selectbox("Pulse shape", ["Rectangular", "Triangular", "Half-sine"])
            Fmax = st.number_input("Peak force Fmax (N)", value=200.0, step=10.0)
            duration = st.number_input("Contact duration (s)", min_value=0.002, value=0.040, step=0.002, format="%.3f")
            pre  = st.number_input("Pre-roll time (s)", min_value=0.0, value=0.050, step=0.010, format="%.3f")
            post = st.number_input("Post-roll time (s)", min_value=0.0, value=0.150, step=0.010, format="%.3f")
            t, F = build_preset_force(shape, Fmax, duration, pre, post, dt)
            editable_df = None
        else:
            total_T = st.number_input("Total time window T (s)", min_value=0.02, value=0.300, step=0.010, format="%.3f")
            st.markdown("Edit the control points below (piecewise **linear**).")
            default_pts = pd.DataFrame({"t": [0.00, 0.05, 0.07, total_T], "F": [0.0, 250.0, 0.0, 0.0]})
            editable_df = st.data_editor(default_pts, num_rows="dynamic", hide_index=True, key="ft_points")
            t, F, editable_df = build_custom_force(editable_df, total_T, dt)

        results = impulse_and_kinematics(t, F, m, v0)
        J_total, v, x = results["J_total"], results["v"], results["x"]

        st.subheader("Outputs")
        st.write(f"**Impulse (total)**  J = `{J_total:.4f}` N·s")
        st.write(f"**Momentum change**  Δp = `{J_total:.4f}` kg·m/s   →   **vₓ final** = `{(m*v0 + J_total)/m:.4f}` m/s")
        st.write(f"**Average force over contact**  F_avg = `{results['F_avg']:.2f}` N")
        if editable_df is not None:
            with st.expander("Control points used (custom mode)"):
                st.dataframe(editable_df, hide_index=True, use_container_width=True)

    with colR:
        st.subheader("Force–time (area is impulse)")
        figF = plot_force_with_area(t, F, J_total)
        st.pyplot(figF, use_container_width=True)

        st.subheader("Velocity–time")
        figV = plot_velocity(t, v, v0, J_total, m)
        st.pyplot(figV, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Impulse–Momentum", layout="wide")
    app()
