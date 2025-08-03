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
  "object": "",
  "motion_type": "",
  "question_type": "",
  "mass": null,
  "initial_velocity": null,
  "final_velocity": null,
  "angle": null,
  "height": null,
  "distance": null,
  "time": null,
  "forces": [
    {{
      "type": "",
      "magnitude": null,
      "direction": ""
    }}
  ],
  "friction_coefficient": null,
  "friction_type": "",
  "spring_constant": null,
  "equilibrium_length": null,
  "radius": null,
  "center_of_mass": null,
  "moment_of_inertia": null,
  "angular_velocity": null,
  "angular_acceleration": null,
  "torque": null,
  "power": null,
  "work": null,
  "energy_type": "",
  "charge": null,
  "current": null,
  "voltage": null,
  "resistance": null,
  "other_objects": [],
  "constraints": "",
  "free_body_diagram": true,
  "notes": ""
}}
Only output the JSON object. Do not include code fences, explanations, or extra text.

Problem: {problem_text}
"""

        response = model.generate_content(prompt)
        content = response.text.strip()

        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
