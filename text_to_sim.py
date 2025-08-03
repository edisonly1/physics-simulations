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
  "object": "...",
  "motion_type": "...",  # e.g. 'projectile', 'free fall', 'inclined'
  "question_type": "...", # e.g. 'acceleration', 'range', 'time', etc.
  "mass": null,
  "initial_velocity": null,
  "angle": null,
  "height": 0,
  "forces": [],
  "question_type": "..."
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
