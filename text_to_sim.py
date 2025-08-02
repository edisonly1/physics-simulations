import google.generativeai as genai
import streamlit as st
import json

def extract_physics_info(problem_text):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    try:
        model = genai.GenerativeModel(model_name="models/gemini-pro")

        prompt = f"""
You are a physics tutor for AP Physics 1.

Extract the following fields from the problem and return only valid JSON:

{{
  "object": "...",
  "motion_type": "...",
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

        # Remove markdown formatting if present
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
