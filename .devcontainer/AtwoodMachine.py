# AtwoodMachine.py
# Streamlit simulator for Full & Half Atwood Machines
# Edison-ready: clean UI, equations shown, simple animation with stop conditions

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

G_DEFAULT = 9.8

def _accel_full(m1, m2, g, use_inertia=False, pulley_mass=0.0, pulley_radius=0.1):
    """
    Full Atwood (two hanging masses). If use_inertia, treat pulley as solid disk:
        I = (1/2) M R^2, a = (m2 - m1) g / (m1 + m2 + I/R^2)
    Returns (a, T_ideal_or_T1, T2_or_None)
    """
    if not use_inertia:
        # Ideal: single tension
        a = (m2 - m1) * g / (m1 + m2)
        T = m1 * (g + a)  # same as m2*(g - a)
        return a, T, None
    else:
        I = 0.5 * pulley_mass * pulley_radius**2
        denom = m1 + m2 + I / (pulley_radius**2)
        a = (m2 - m1) * g / denom
        # Different tensions each side
        T1 = m1 * (g + a)          # lighter side goes up if m2>m1
        T2 = m2 * (g - a)
        return a, T1, T2

def _accel_half(m_table, m_hang, g, mu_k=0.0, use_inertia=False, pulley_mass=0.0, pulley_radius=0.1):
    """
    Half Atwood (cart on horizontal + hanging mass).
    If friction on table: subtract mu*m_table*g from driving force (assuming motion is pulled by hanging mass).
    If use_inertia, add I/R^2 to effective mass in denominator.
    a = (m_hang*g - mu*m_table*g) / (m_table + m_hang + I/R^2)
    If numerator <= 0 -> a = 0 (won't move).
    Returns (a, T_ideal_or_T_cart, T_hang_or_None)
    """
    I_term = (0.5 * pulley_mass * pulley_radius**2) / (pulley_radius**2) if use_inertia else 0.0
    drive = m_hang * g - mu_k * m_table * g
    if drive <= 0:
        return 0.0, mu_k * m_table * g, None  # static-ish stop (simple)
    a = drive / (m_table + m_hang + I_term)

    if not use_inertia:
        T = m_table * a + mu_k * m_table * g  # tension felt by cart side
        return a, T, None
    else:
        # With inertia, tensions differ slightly; compute from each mass equation
        T_cart = m_table * a + mu_k * m_table * g
        T_hang = m_hang * (g - a)
        return a, T_cart, T_hang

def _equation_block_full(use_inertia):
    if not use_inertia:
        st.latex(r"a=\frac{(m_2-m_1)g}{m_1+m_2},\quad T=m_1(g+a)=m_2(g-a)")
    else:
        st.latex(r"I=\tfrac12 M_p R^2,\quad a=\frac{(m_2-m_1)g}{m_1+m_2+\frac{I}{R^2}}")
        st.latex(r"T_1=m_1(g+a),\quad T_2=m_2(g-a)")

def _equation_block_half(use_inertia):
    if not use_inertia:
        st.latex(r"a=\frac{m_h g-\mu_k m_t g}{m_t+m_h},\quad T=m_t a+\mu_k m_t g")
    else:
        st.latex(r"I=\tfrac12 M_p R^2,\quad a=\frac{m_h g-\mu_k m_t g}{m_t+m_h+\frac{I}{R^2}}")
        st.latex(r"T_{\text{cart}}=m_t a+\mu_k m_t g,\quad T_{\text{hang}}=m_h(g-a)")

def _anim_header():
    left, right = st.columns([1,1])
    with left:
        fps = st.slider("FPS", 10, 60, 30, 1)
    with right:
        sim_time = st.slider("Max sim time (s)", 2.0, 20.0, 8.0, 0.5)
    return fps, sim_time

def _animate_full(a, y0_left=0.6, y0_right=0.2, pulley_y=0.9, floor_y=0.05,
                  fps=30, sim_time=6.0, scale=0.15, direction_down_right=True):
    """
    Simple 2D schematic: left and right masses on either side of a pulley.
    Stops if a mass hits floor or time ends.
    """
    dt = 1.0 / fps
    # choose sign so that if a>0 then right mass goes down (m2>m1 typical)
    sgn = 1 if direction_down_right else -1

    # Rope length -> displacements equal magnitude and opposite sign
    t = 0.0
    v = 0.0
    s = 0.0

    ph = st.empty()
    while t <= sim_time:
        s = 0.5 * sgn * a * t**2
        # positions (clamp at floor/top)
        y_left = np.clip(y0_left + s, floor_y + scale/2, pulley_y - scale/2)
        y_right = np.clip(y0_right - s, floor_y + scale/2, pulley_y - scale/2)

        # Stop if either hits limit
        if (y_left <= floor_y + scale/2 + 1e-4) or (y_right <= floor_y + scale/2 + 1e-4):
            pass_flag = True
        else:
            pass_flag = False

        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Pulley
        pulley_x = 0.5
        circle = plt.Circle((pulley_x, pulley_y), 0.05, fill=False, lw=2)
        ax.add_patch(circle)

        # Rope lines
        ax.plot([pulley_x, 0.2], [pulley_y, y_left + scale/2], lw=2)
        ax.plot([pulley_x, 0.8], [pulley_y, y_right + scale/2], lw=2)

        # Mass blocks
        rect1 = plt.Rectangle((0.15, y_left - scale/2), 0.1, scale, ec='k', fc='lightgray')
        rect2 = plt.Rectangle((0.75, y_right - scale/2), 0.1, scale, ec='k', fc='lightgray')
        ax.add_patch(rect1); ax.add_patch(rect2)

        # Floor
        ax.plot([0,1],[floor_y,floor_y],'k-', lw=2)

        ax.text(0.15, y_left + scale + 0.02, "m₁", fontsize=12, ha='center')
        ax.text(0.85, y_right + scale + 0.02, "m₂", fontsize=12, ha='center')
        ax.text(0.02, 0.96, f"t = {t:0.2f} s", fontsize=10)

        ph.pyplot(fig, clear_figure=True)
        plt.close(fig)

        if pass_flag:
            break

        time.sleep(dt)
        t += dt

def _animate_half(a, x0=0.15, y0=0.5, edge_x=0.85, floor_y=0.05,
                  fps=30, sim_time=6.0, cart_w=0.12, cart_h=0.06):
    """
    Schematic: cart on table (bottom), hanging mass on right side over pulley.
    We map rope displacement equally: x increases to right; y decreases (hangs down).
    Stops if cart reaches edge or mass hits floor.
    """
    dt = 1.0 / fps
    t = 0.0

    ph = st.empty()
    while t <= sim_time:
        s = 0.5 * a * t**2  # displacement pulled by hanging mass

        # positions
        x_cart = np.clip(x0 + s, 0.05 + cart_w/2, edge_x - cart_w/2)
        y_hang = np.clip(y0 - s, floor_y + cart_h/2, 0.95 - cart_h/2)

        hit_edge = (x_cart >= edge_x - cart_w/2 - 1e-4) or (y_hang <= floor_y + cart_h/2 + 1e-4)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.axis("off")

        # Table
        ax.plot([0.05, edge_x], [0.25, 0.25], 'k-', lw=3)
        # Pulley at corner
        pulley_x, pulley_y = edge_x, 0.25
        circ = plt.Circle((pulley_x, pulley_y), 0.04, fill=False, lw=2)
        ax.add_patch(circ)

        # Rope: cart to pulley to hanging mass
        ax.plot([x_cart, pulley_x], [0.25+cart_h/2, pulley_y], lw=2)          # along table
        ax.plot([pulley_x, pulley_x], [pulley_y, y_hang + cart_h/2], lw=2)    # down

        # Cart
        cart = plt.Rectangle((x_cart - cart_w/2, 0.25), cart_w, cart_h, ec='k', fc='lightgray')
        ax.add_patch(cart)
        ax.text(x_cart, 0.25 + cart_h + 0.03, "m₁ (table)", ha='center', fontsize=11)

        # Hanging block
        block = plt.Rectangle((pulley_x - cart_w/2, y_hang - cart_h/2), cart_w, cart_h, ec='k', fc='lightgray')
        ax.add_patch(block)
        ax.text(pulley_x + 0.08, y_hang, "m₂ (hanging)", va='center', fontsize=11)

        # Floor
        ax.plot([pulley_x - 0.1, pulley_x + 0.1],[floor_y, floor_y],'k-', lw=2)

        ax.text(0.02, 0.96, f"t = {t:0.2f} s", fontsize=10)
        ph.pyplot(fig, clear_figure=True)
        plt.close(fig)

        if hit_edge:
            break

        time.sleep(dt)
        t += dt

def app():
    st.title("Atwood Machine — Full & Half")

    mode = st.radio("Choose setup", ["Full Atwood (two hanging masses)", "Half Atwood (cart + hanging mass)"])

    g = st.number_input("g (m/s²)", 5.0, 20.0, G_DEFAULT, 0.1)

    adv = st.checkbox("Advanced options (pulley inertia / friction)", value=False)

    if mode.startswith("Full"):
        col1, col2 = st.columns(2)
        with col1:
            m1 = st.number_input("m₁ (kg) — left", 0.01, 50.0, 4.0, 0.1)
        with col2:
            m2 = st.number_input("m₂ (kg) — right", 0.01, 50.0, 6.0, 0.1)

        pulley_mass = 0.0
        pulley_radius = 0.1
        use_inertia = False
        if adv:
            use_inertia = st.checkbox("Include pulley rotational inertia", value=True)
            pulley_mass = st.number_input("Pulley mass Mₚ (kg)", 0.0, 50.0, 1.0, 0.1)
            pulley_radius = st.number_input("Pulley radius R (m)", 0.01, 1.0, 0.10, 0.01)

        st.subheader("Equations")
        _equation_block_full(use_inertia)

        a, T_left_or_single, T_right = _accel_full(m1, m2, g, use_inertia, pulley_mass, pulley_radius)

        st.markdown("**Results**")
        cols = st.columns(3)
        cols[0].metric("Acceleration |a| (m/s²)", f"{abs(a):.3f}", help="Positive means right mass down in our drawing.")
        if T_right is None:
            cols[1].metric("Tension T (N)", f"{T_left_or_single:.2f}")
        else:
            cols[1].metric("Tension T₁ (left) (N)", f"{T_left_or_single:.2f}")
            cols[2].metric("Tension T₂ (right) (N)", f"{T_right:.2f}")

        st.divider()
        st.subheader("Animation")
        fps, sim_time = _anim_header()
        go = st.button("Run animation", type="primary")
        if go:
            _animate_full(a, fps=fps, sim_time=sim_time, direction_down_right=(m2 >= m1))

        with st.expander("Classroom tips"):
            st.write("- Have students predict direction by comparing \(m_2\) and \(m_1\).")
            st.write("- Toggle pulley inertia to see why real systems accelerate a bit less than the ideal formula.")

    else:
        col1, col2 = st.columns(2)
        with col1:
            m_table = st.number_input("m₁ (kg) — on table", 0.01, 50.0, 5.0, 0.1)
        with col2:
            m_hang = st.number_input("m₂ (kg) — hanging", 0.01, 50.0, 2.0, 0.1)

        mu_k = st.slider("Coefficient of kinetic friction μₖ (table)", 0.0, 1.0, 0.0, 0.01)

        pulley_mass = 0.0
        pulley_radius = 0.1
        use_inertia = False
        if adv:
            use_inertia = st.checkbox("Include pulley rotational inertia", value=True)
            pulley_mass = st.number_input("Pulley mass Mₚ (kg)", 0.0, 50.0, 1.0, 0.1)
            pulley_radius = st.number_input("Pulley radius R (m)", 0.01, 1.0, 0.10, 0.01)

        st.subheader("Equations")
        _equation_block_half(use_inertia)

        a, T_cart_or_single, T_hang = _accel_half(m_table, m_hang, g, mu_k, use_inertia, pulley_mass, pulley_radius)

        st.markdown("**Results**")
        cols = st.columns(3)
        cols[0].metric("Acceleration |a| (m/s²)", f"{abs(a):.3f}", help="Positive means cart → right, hanger ↓.")
        if T_hang is None:
            cols[1].metric("Tension T (N)", f"{T_cart_or_single:.2f}")
        else:
            cols[1].metric("Tension (cart side) (N)", f"{T_cart_or_single:.2f}")
            cols[2].metric("Tension (hanging side) (N)", f"{T_hang:.2f}")

        st.divider()
        st.subheader("Animation")
        fps, sim_time = _anim_header()
        go = st.button("Run animation", type="primary")
        if go:
            _animate_half(a, fps=fps, sim_time=sim_time)

        with st.expander("Classroom tips"):
            st.write("- Set μₖ high and show how friction can prevent motion (numerator ≤ 0 ⇒ a = 0).")
            st.write("- Compare ideal vs. inertia to discuss energy going into pulley rotation.")
