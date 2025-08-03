import google.generativeai as genai
import streamlit as st
import json

def extract_physics_info(problem_text):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        prompt = f"""
You are an AP Physics 1 tutor.

Extract only the physics parameters from this word problem and return them as a JSON object in this format:

{{
  "object": "...",  // e.g. 'ball', 'block', 'cart'
  "motion_type": "...",  // 'projectile', 'free fall', 'inclined', 'circular', 'oscillation', 'static', etc.
  "question_type": "...", // 'acceleration', 'range', 'time', 'force', 'energy', 'work', etc.
  
  "mass": null,
  "initial_velocity": null,
  "final_velocity": null,
  "angle": null,
  "height": null,
  "distance": null,      // for displacement/travelled, etc.
  "time": null,

  "forces": [            // list of forces acting on object(s)
    // Each force as an object for clarity
    {
      "type": "...",     // e.g. 'gravity', 'normal', 'friction', 'tension', 'applied', 'spring', 'air_resistance'
      "magnitude": null,
      "direction": "..." // e.g. 'up', 'down', 'left', 'right', 'along incline', degrees, etc.
    }
  ],

  "friction_coefficient": null, // μ, for kinetic or static as needed
  "friction_type": "...",       // 'static' or 'kinetic'
  "spring_constant": null,      // k, for spring/block problems
  "equilibrium_length": null,   // For springs

  "radius": null,       // For circular motion
  "center_of_mass": null, // If specified or relevant
  "moment_of_inertia": null, // For rotational dynamics

  "angular_velocity": null,
  "angular_acceleration": null,
  "torque": null,

  "power": null,
  "work": null,
  "energy_type": "...", // 'kinetic', 'potential', 'thermal', etc.

  "charge": null,       // For circuit or electrostatics overlap
  "current": null,
  "voltage": null,
  "resistance": null,

  "other_objects": [],   // List of other objects, if multi-object (include same schema inside)

  "constraints": "...", // e.g. 'frictionless', 'rough', 'at rest', 'constant speed', etc.
  "diagram_requested": true, // If a diagram is needed in the output

  "notes": "..."        // Any extra parsed details or context
}}


Problem: {problem_text}
"""

        response = model.generate_content(prompt)
        content = response.text.strip()

        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
