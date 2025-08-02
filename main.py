import streamlit as st
import ProjectileMotion
from motion_parser import parse_problem

st.sidebar.title("Physics Simulation Lab")
page = st.sidebar.selectbox("Choose a Simulation", [
    "Home",
    "Projectile Motion (Manual)",
    "Projectile Motion (AI Word Problem)"
])

if page == "Home":
    st.title("Welcome to AP Physics Simulator")
    st.markdown("Use the sidebar to explore different simulations.")

elif page == "Projectile Motion (Manual)":
    ProjectileMotion.app()

elif page == "Projectile Motion (AI Word Problem)":
    st.title("AI Word Problem Interpreter")
    problem = st.text_area("Paste an AP Physics 1-style word problem:")

    if st.button("Extract and Simulate"):
        parsed = parse_problem(problem)
        st.write("Parsed Data:", parsed)

        v0 = parsed.get("initial_velocity")
        angle = parsed.get("angle")
        height = parsed.get("height", 0)

        if v0 and angle is not None:
            from ProjectileMotion import simulate_projectile
            x, y, t_flight = simulate_projectile(v0, angle, height)

            st.markdown(f"**Time of Flight:** {t_flight:.2f} s")
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Height (m)")
            ax.set_title("Trajectory")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.error("Could not extract velocity or angle. Please revise.")
