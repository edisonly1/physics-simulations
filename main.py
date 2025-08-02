import streamlit as st
import ProjectileMotion

st.sidebar.title("🧪 Physics Simulation Lab")
page = st.sidebar.selectbox("Choose a Simulation", ["Home", "Projectile Motion"])

if page == "Home":
    st.title("👋 Welcome!")
    st.markdown("Use the sidebar to explore simulations.")
elif page == "Projectile Motion":
    ProjectileMotion.app()
