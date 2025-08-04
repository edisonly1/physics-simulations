import streamlit as st
import ProjectileMotion
from text_to_sim import extract_physics_info, extract_solution_steps
import InclinedPlane
import FreeFall

# Sidebar Navigation
st.sidebar.title("Physics Simulation Lab")
page = st.sidebar.selectbox("Choose a Simulation", ["Home", "AI Problem Parser"])

# Page Routing
if page == "Home":
    st.title("Welcome to the Physics Simulation Lab!")
    st.markdown("""
    This tool is designed to help visualize AP Physics 1 problems.
    
    - Use AI to break down word problems
    - More simulations coming soon!
    
    Select the problem parser from the sidebar to get started.
    """)

elif page == "AI Problem Parser":
    st.title("AI-Powered Problem Interpreter")
    st.markdown("Paste in an AP Physics style word problem and let Gemini extract the physical setup.")

    problem = st.text_area("Enter your physics problem:")

    # Store problem in session state so it's available later
    if problem:
        st.session_state["problem_text"] = problem

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
    problem_text = st.session_state.get("problem_text", "")

    if parsed and "initial_velocity" in parsed and "angle" in parsed:
        if st.button("Simulate This Problem"):
            motion_type = parsed.get("motion_type", "").lower()
            constraints = parsed.get("constraints", "").lower()
            notes = parsed.get("notes", "").lower()

            # Show simulation (match your logic for routing)
            if motion_type == "projectile":
                ProjectileMotion.app(data=parsed)
            elif motion_type in ["free fall"]:
                FreeFall.app(data=parsed)
            elif "incline" in constraints or "incline" in notes:
                InclinedPlane.app(data=parsed)
            else:
                st.warning(f"Simulation for motion type '{motion_type}' not implemented.")

            # Get AI solution steps (Gemini) and display after simulation
            st.markdown("### Step-by-step Solution")
            with st.spinner("Generating solution steps..."):
                steps = extract_solution_steps(problem_text)
                st.markdown(steps)
