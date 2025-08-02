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
  "mass": null,
  "angle": null,
  "initial_velocity": null,
  "height": 0,
  "forces": [],
  "question_type": ""
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
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    return response.json()
