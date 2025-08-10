# ImpulseMomentum.py
# Streamlit module: Impulse–Momentum Theorem (AP Physics 1)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- math helpers ----------
TOL = 1e-9

def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum((y[1:] + y[:-1]) * 0.5 * np.diff(x))
    return out

def build_preset_force(shape: str, Fmax: float, duration: float,
                       pre: float, post: float, dt: float):
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
        if np.any(up):
            F[up] = Fmax * (t[up] - t0) / max(TOL, (mid - t0))
        if np.any(dn):
            F[dn] = Fmax * (t1 - t[dn]) / max(TOL, (t1 - mid))
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
    if df["t"].iloc[0] > 0.0:
        df = pd.concat([pd.DataFrame([{"t": 0.0, "F": 0.0}]), df], ignore_index=True)
    if df["t"].iloc[-1] < total_time:
        df = pd.concat([df, pd.DataFrame([{"t": total_time, "F": 0.0}])], ignore_index=True)

    t = np.linspace(0.0, total_time, max(200, int(round(total_time/dt))))
    F = np.interp(t, df["t"].to_numpy(), df["F"].to_numpy())
    return t, F, df

def impulse_and_kinematics(t: np.ndarray, F: np.ndarray, m: float, v0: float):
    J_total = float(np.trapz(F, t))
    mask = np.abs(F) > TOL
    if np.count_nonzero(mask) >= 2:
        t_contact = t[mask]
        F_contact = F[mask]
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
    return {"J_total": J_total, "J_contact": J_contact, "F_avg": F_avg,
            "v": v, "x": x, "p": p, "a": a}

# ---------- plots ----------
def plot_force_with_area(t, F, J):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, F, lw=2, label="Force $F(t)$")
    ax.fill_between(t, 0, F, alpha=0.25, label=f"Impulse area  J = {J:.3f} N·s")
    ax.axhline(0, lw=1, color="#888888"); ax.set_xlabel("time (s)"); ax.set_ylabel("force (N)")
    ax.legend(loc="best"); ax.grid(alpha=0.25); fig.tight_layout(); return fig

def plot_velocity(t, v, v0, J, m):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, v, lw=2, label="Velocity $v(t)$")
    ax.axhline(v0, ls="--", lw=1, label=f"$v_0$ = {v0:.3f} m/s")
    ax.set_xlabel("time (s)"); ax.set_ylabel("velocity (m/s)"); ax.grid(alpha=0.25)
    ax.legend(loc="best", title=f"$\\Delta v$ = J/m = {J/m:.3f} m/s"); fig.tight_layout(); return fig

def draw_ball_panel(t, x, v, t_now):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    x_span = max(1.0, np.max(x) - np.min(x)); pad = 0.1 * x_span
    ax.set_xlim(np.min(x) - pad, np.max(x) + pad); ax.set_ylim(-1, 1); ax.set_yticks([])
    ax.set_xlabel("position (m)  —  speed shown by trail length"); ax.grid(alpha=0.15, axis="x")
    idx = max(0, int(np.searchsorted(t, t_now, side="right") - 1))
    x_now = x[idx]
    trail_len = 0.15 * (np.abs(v[idx]) / (np.max(np.abs(v)) + TOL)) * (np.max(x) - np.min(x) + TOL)
    ax.plot([x_now - trail_len, x_now], [0, 0], lw=5, alpha=0.35)
    ax.scatter([x_now], [0], s=400, edgecolor="k", linewidths=1.2)
    fig.tight_layout(); return fig

# ---------- Streamlit UI ----------
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

        # compute
        results = impulse_and_kinematics(t, F, m, v0)
        J, v, x = results["J_total"], results["v"], results["x"]

        st.subheader("Outputs")
        st.write(f"**Impulse (total)**  J = `{J:.4f}` N·s")
        st.write(f"**Momentum change**  Δp = `{J:.4f}` kg·m/s   →   **vₓ final** = `{(m*v0 + J)/m:.4f}` m/s")
        st.write(f"**Average force over contact**  F_avg = `{results['F_avg']:.2f}` N")
        if editable_df is not None:
            with st.expander("Control points used (custom mode)"):
                st.dataframe(editable_df, hide_index=True, use_container_width=True)

    with colR:
        st.subheader("Force–time (area is impulse)")
        st.pyplot(plot_force_with_area(t, F, J), use_container_width=True)

        st.subheader("Velocity–time")
        st.pyplot(plot_velocity(t, v, v0, J, m), use_container_width=True)

        # -------- Animation controls (Play/Pause) --------
        st.subheader("Animation: ball struck by force profile")

        # initialize persistent state
        t_min, t_max = float(t[0]), float(t[-1])
        step_time = float(max(dt, (t_max - t_min) / 200.0))
        if "imp_t_now" not in st.session_state: st.session_state.imp_t_now = t_min
        if "imp_is_playing" not in st.session_state: st.session_state.imp_is_playing = False
        if "imp_speed" not in st.session_state: st.session_state.imp_speed = 1.0

        # clamp if user changed parameters/time window
        st.session_state.imp_t_now = float(np.clip(st.session_state.imp_t_now, t_min, t_max))

        c1, c2, c3 = st.columns([1, 1, 2])
        if not st.session_state.imp_is_playing:
            if c1.button("▶ Play", use_container_width=True):
                st.session_state.imp_is_playing = True
                st.rerun()
        else:
            if c1.button("⏸ Pause", use_container_width=True):
                st.session_state.imp_is_playing = False
                st.experimental_rerun()

        if c2.button("⟲ Reset", use_container_width=True):
            st.session_state.imp_t_now = t_min
            st.session_state.imp_is_playing = False
            st.experimental_rerun()

        st.session_state.imp_speed = c3.select_slider(
            "Speed", options=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0], value=st.session_state.imp_speed
        )

        # scrubber (manual)
        t_now = st.slider("Scrub time",
                          min_value=t_min, max_value=t_max,
                          value=float(st.session_state.imp_t_now),
                          step=step_time, format="%.3f")
        st.session_state.imp_t_now = float(t_now)

        # draw current frame
        st.pyplot(draw_ball_panel(t, x, v, st.session_state.imp_t_now), use_container_width=True)

        # auto-advance when playing
        if st.session_state.imp_is_playing:
            time.sleep(0.016)  # ~60 FPS
            next_t = st.session_state.imp_t_now + st.session_state.imp_speed * step_time
            if next_t >= t_max:
                st.session_state.imp_t_now = t_max
                st.session_state.imp_is_playing = False
            else:
                st.session_state.imp_t_now = float(next_t)
            st.rerun()

        st.markdown(
            "*Notes:* The shaded area under $F(t)$ is the impulse $J$. "
            "Velocity changes by $\\Delta v = J/m$. The ball’s position reflects the integral of $v(t)$."
        )

if __name__ == "__main__":
    st.set_page_config(page_title="Impulse–Momentum", page_icon="🟢", layout="wide")
    app()
