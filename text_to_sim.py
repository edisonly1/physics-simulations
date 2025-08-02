# text_to_sim.py

import streamlit as st
import requests

def extract_physics_info(prompt_text):
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"
    }

    # Simpler prompt format (FLAN models don't do well with JSON output)
    prompt = f"""
Given this AP Physics 1 problem:

"{prompt_text}"

Extract and identify the following:
- Object
- Initial velocity
- Angle
- Type of motion
- What the question is asking for

Respond like:
Object: ...
Initial Velocity: ...
Angle: ...
Motion Type: ...
Question Type: ...
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.5
        }
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            st.error(f"Hugging Face API error: {response.status_code}")
            st.code(response.text)
            return {"error": "Model request failed."}

        output = response.json()

        # FLAN-T5 returns a string inside a list of dicts
        if isinstance(output, list) and "generated_text" in output[0]:
            result = output[0]["generated_text"].strip()
            return {"extracted": result}
        else:
            st.error("Unexpected model output format.")
            st.json(output)
            return {"error": "Unexpected model output."}

    except Exception as e:
        st.error(f"Exception occurred: {e}")
        return {"error": str(e)}
