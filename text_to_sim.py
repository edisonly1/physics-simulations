import google.generativeai as genai
import streamlit as st
import json

def extract_physics_info(problem_text):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    try:
        model = genai.GenerativeModel("gemini-pro")  # or "gemini-1.5-pro" if you have access

        prompt = f"""
You are a physics tutor for AP Physics 1.

Extract and return only this JSON object based on the problem:

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

        # Clean markdown if present
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
