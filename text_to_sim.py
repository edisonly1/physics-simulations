from google import generativeai as genai
import streamlit as st
import json

def extract_physics_info(problem_text):
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

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
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Or "gemini-1.5-pro" if available
            contents=prompt
        )

        content = response.text.strip()

        # Remove markdown block if present
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
