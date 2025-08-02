# text_to_sim.py

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

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers=headers,
            json=payload
        )

        # Check for non-200 response
        if response.status_code != 200:
            st.error(f"Hugging Face API error: {response.status_code}")
            st.code(response.text)
            return {"error": "Model request failed."}

        output = response.json()

        # Sometimes Hugging Face returns just a string, sometimes a list
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        else:
            st.error("Unexpected model output format.")
            st.json(output)
            return {"error": "Unexpected model output."}

    except Exception as e:
        st.error(f"Exception occurred: {e}")
        return {"error": str(e)}
