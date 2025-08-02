import google.generativeai as genai
import streamlit as st
import json

def extract_physics_info(problem_text):
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
You are a physics tutor for AP Physics 1.

Your task is to extract the key physics quantities from the following problem and return them as a JSON object with these fields:

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

Only return a valid JSON object. Do not include explanation.

Problem: {problem_text}
"""

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Try parsing it as JSON (in case Gemini wraps it in Markdown)
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
