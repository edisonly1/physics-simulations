import streamlit as st
import ProjectileMotion
import InclinedPlane
import FreeFall
import UniformLinearAccelerated1D as U1D 
import NewtonsSecondLaw
import FrictionModels
import AtwoodMachine
import SpingMass
import WorkEnergy

st.set_page_config(page_title="Physics Simulation Lab", layout="wide")

# ---- Sidebar ----
st.sidebar.title("Physics Simulation Lab")
page = st.sidebar.selectbox(
    "Choose a Simulation",
    [
        "Home",
        "Uniform Motion (1D)",      
        "Projectile Motion",
        "Inclined Plane",
        "Free Fall",
        "Newton's Second Law",
        "Friction Models",
        "Atwood Machine",
        "Spring-Mass System",
        "Work-Energy Theorem",
    ],
)

# ---- Pages ----
if page == "Home":
    st.title("Welcome to the Physics Simulation Lab! 🧪")
    st.markdown("""
This is a classroom-friendly simulator with variable selectors.

**Included modules**
- **Uniform Motion (1D)** — constant velocity & constant acceleration; p–t, v–t, a–t graphs; overlays
- **Projectile Motion** — launch at any angle and/or from a height; animated path + formulas
- **Inclined Plane** — friction, up/down cases, stopping distance; FBD viewer + animation
- **Free Fall** — 1D vertical motion; time to ground + animation

**How to use**
1. Pick a simulation from the sidebar.
2. Adjust sliders and inputs.
3. Read the key results and expand the formulas/graphs where available.
""")

elif page == "Uniform Motion (1D)":  
    st.title("Uniform Linear & Accelerated Motion (1D)")
    U1D.app()

elif page == "Projectile Motion":
    st.title("Projectile Motion Simulator")
    ProjectileMotion.app()  # interactive mode (no data)

elif page == "Inclined Plane":
    st.title("Inclined Plane Simulator")
    InclinedPlane.app()     # interactive mode (no data)

elif page == "Free Fall":
    st.title("Free Fall Simulator")
    FreeFall.app()          # interactive mode (no data)

elif page == "Newton's Second Law":
    st.title("Newton's Second Law Simulation")
    NewtonsSecondLaw.app()

elif page == "Friction Models":
    FrictionModels.app()

elif page == "Atwood Machine":
    AtwoodMachine.app()

elif page == "Spring-Mass System":
    SpingMass.app()

elif page == "Work-Energy Theorem":
    WorkEnergy.app()