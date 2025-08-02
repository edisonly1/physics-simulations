import streamlit as st
import ProjectileMotion
from text_to_sim import extract_physics_info

st.sidebar.title("Physics Simulation Lab")
page = st.sidebar.selectbox("Choose a Simulation", ["Home", "Projectile Motion", "AI Problem"])

if page == "Home":
    st.title("Welcome!")
    st.markdown("Use the sidebar to explore simulations.")
elif page == "Projectile Motion":
    ProjectileMotion.app()
elif page == "AI Problem Parser":
    st.title("AI-Powered Problem Interpreter")

    st.markdown("Paste an AP Physics 1-style word problem and let AI break it down into physics quantities and concepts.")

    problem = st.text_area("Enter your physics problem:")
    
    if st.button("Extract Physics Info"):
        with st.spinner("Thinking..."):
            try:
                parsed = extract_physics_info(problem)
                st.subheader("Parsed Physics Setup")
                st.json(parsed)
            except Exception as e:
                st.error(f"Something went wrong: {e}")