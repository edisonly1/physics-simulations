# text_to_sim.py

import streamlit as st
import requests

def extract_physics_info(prompt_text):
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"
    }

    payload = {
        "inputs": f"""You are a physics tutor for AP Physics 1.

Given this problem: {prompt_text}

Extract the scenario as structured JSON with fields:
- object name
- motion type
- mass
- angle (if relevant)
- initial velocity
- forces (if relevant)
- question type (acceleration, velocity, etc.)""",
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        headers=headers,
        json=payload
    )

    output = response.json()
    
    # Return the raw text for now — we'll parse it later
    return output[0]["generated_text"]
