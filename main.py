import streamlit as st
import ProjectileMotion
from text_to_sim import extract_physics_info

# Sidebar Navigation
st.sidebar.title("Physics Simulation Lab")
page = st.sidebar.selectbox("Choose a Simulation", ["Home", "Projectile Motion", "AI Problem Parser"])

# Page Routing
if page == "Home":
    st.title("Welcome to the Physics Simulation Lab!")
    st.markdown("""
    This tool is designed to help visualize AP Physics 1 problems.
    
    - Use AI to break down word problems
    - Explore projectile motion with graphs and sliders
    - More simulations coming soon!
    
    Select a simulation from the sidebar to get started.
    """)

elif page == "Projectile Motion":
    ProjectileMotion.app()  # Assuming this is defined in ProjectileMotion.py

elif page == "AI Problem Parser":
    st.title("AI-Powered Problem Interpreter")
    st.markdown("Paste in an AP Physics style word problem and let Gemini extract the physical setup.")

    problem = st.text_area("Enter your physics problem:")

    if st.button("Extract Physics Info"):
        if problem.strip() == "":
            st.warning("Please enter a problem first.")
        else:
            with st.spinner("Analyzing with Gemini..."):
                result = extract_physics_info(problem)
                st.session_state["parsed_result"] = result  # Store for later use
                st.subheader("Parsed Physics Setup")
                st.json(result)

    # Load from session state (after user hits "Extract")
    parsed = st.session_state.get("parsed_result", None)

    if parsed and "initial_velocity" in parsed and "angle" in parsed:
        if st.button("Simulate This Problem"):
            ProjectileMotion.app(
                velocity=parsed["initial_velocity"],
                angle=parsed["angle"],
                height=parsed.get("height", 0)
            )
