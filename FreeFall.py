import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app(data=None):
    st.title("Free Fall Simulator")
    if data:
        v0 = data.get("initial_velocity", 0)
        h0 = data.get("height", 0)
        st.markdown("#### Using AI-extracted values:")
        st.markdown(f"- **Initial Height:** `{h0} m`")
        st.markdown(f"- **Initial Velocity:** `{v0} m/s`")
    else:
        h0 = st.slider("Initial Height (m)", 0.0, 100.0, 10.0)
        v0 = st.slider("Initial Velocity (m/s)", 0.0, 20.0, 0.0)

    g = 10.0  # m/s²

    # Time to fall
    t_fall = (-v0 + np.sqrt(v0**2 + 2 * g * h0)) / g
    t = np.linspace(0, t_fall, 300)
    y = h0 + v0 * t - 0.5 * g * t**2

    # Display results
    st.markdown("### Results")
    st.markdown(f"- **Time to hit ground:** `{t_fall:.2f}` seconds")
    st.markdown(f"- **Final speed on impact:** `{(v0 + g * t_fall):.2f}` m/s")

    with st.expander("View Calculations and Formulas"):
        st.markdown(r"""
        **Equations Used:**

        - Displacement: $y(t) = h_0 + v_0 t - \frac{1}{2} g t^2$
        - Time of fall: $t = \frac{-v_0 + \sqrt{v_0^2 + 2gh_0}}{g}$
        - Final speed: $v = v_0 + gt$
        """)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Vertical Motion: Free Fall")
    ax.grid(True)
    st.pyplot(fig)
