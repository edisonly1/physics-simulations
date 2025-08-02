import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app():
    st.title("Projectile Motion Simulator")
    v0 = st.slider("Initial Velocity (m/s)", 0.0, 50.0, 25.0, step=0.1)
    angle_deg = st.slider("Launch Angle (degrees)", 0.0, 90.0, 45.0, step=0.1)
    height = st.slider("Initial Height (m)", 0.0, 10.0, 0.0, step=0.1)

    g = 9.8
    angle = np.radians(angle_deg)
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)

    t_flight = (vy + np.sqrt(vy**2 + 2 * g * height)) / g
    t = np.linspace(0, t_flight, 300)
    x = vx * t
    y = height + vy * t - 0.5 * g * t**2

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    st.pyplot(fig)
