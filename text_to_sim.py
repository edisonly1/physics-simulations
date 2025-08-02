import streamlit as st
import requests

def extract_physics_info(prompt_text):
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"
    }

    prompt = f"""
You are a physics tutor for AP Physics 1.

Given this problem: "{prompt_text}"

Extract the scenario as structured JSON with fields:
{{
  "object": "ball",
  "motion_type": "projectile",
  "mass": 1.2,
  "angle": 32,
  "initial_velocity": 12,
  "height": 0,
  "forces": [],
  "question_type": "horizontal and vertical velocity components"
}}

Only output the JSON object. Do not explain anything.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.5
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise RuntimeError(response.json())

    return response.json()
