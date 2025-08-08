# main.py — AP Physics 1 Simulator (no parser)
import streamlit as st
import ProjectileMotion
import InclinedPlane
import FreeFall

st.set_page_config(page_title="Physics Simulation Lab", page_icon="📐", layout="wide")

# ---- Sidebar ----
st.sidebar.title("Physics Simulation Lab")
page = st.sidebar.selectbox(
    "Choose a Simulation",
    ["Home", "Projectile Motion", "Inclined Plane", "Free Fall"],
)

# ---- Pages ----
if page == "Home":
    st.title("Welcome to the Physics Simulation Lab! 🧪")
    st.markdown("""
This is a **parser-free**, classroom-friendly simulator with variable selectors only.

**Included modules**
- **Projectile Motion** — launch at any angle and/or from a height; animated path + formulas
- **Inclined Plane** — friction, up/down cases, stopping distance; FBD viewer + animation
- **Free Fall** — 1D vertical motion; time to ground + animation

**How to use**
1. Pick a simulation from the sidebar.
2. Adjust sliders and inputs.
3. Read the key results and expand the formulas/graphs where available.

> Tip: If you ever want the AI parser back later, we can add it behind a toggle without touching the simulators.
""")

elif page == "Projectile Motion":
    st.title("Projectile Motion Simulator")
    ProjectileMotion.app()  # interactive mode (no data)

elif page == "Inclined Plane":
    st.title("Inclined Plane Simulator")
    InclinedPlane.app()     # interactive mode (no data)

elif page == "Free Fall":
    st.title("Free Fall Simulator")
    FreeFall.app()          # interactive mode (no data)
