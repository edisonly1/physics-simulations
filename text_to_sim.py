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
  "length": null,
  "angle": null,
  "direction": "",
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

def extract_solution_steps(problem_text):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
You are an expert AP Physics 1 tutor.

Given the following problem, write a detailed, step-by-step solution for a high school student. 
- Use Markdown for lists and structure.
- For equations, use single dollar signs `$...$` for inline math, and double dollar signs `$$...$$` for display/block math.
- Do NOT use triple backticks (```) or code blocks for equations.
- Do not generate a free body diagram or any extra words. Only the steps and explanations for the following problem.
- Use $g = 10$ m/s² for gravity.

Problem:
\"\"\"
{problem_text}
\"\"\"
"""
        response = model.generate_content(prompt)
        content = response.text.strip()
        content = content.replace('```', '$$').replace('\\\\', '\\')
        return content
    except Exception as e:
        return f"**Error generating solution:** {e}"
