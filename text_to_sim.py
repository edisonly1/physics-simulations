import openai
import streamlit as st
import json

def extract_physics_info(problem_text):
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    system_prompt = """
You are a helpful AP Physics 1 tutor.

Your task is to extract key physics quantities and motion types from a word problem.

Return only a JSON object with the following fields:
{
  "object": "...",
  "motion_type": "...",
  "mass": null,
  "initial_velocity": null,
  "angle": null,
  "height": 0,
  "forces": [],
  "question_type": "..."
}
Only output the JSON. Do not explain anything.
"""

    user_prompt = f"Problem: {problem_text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        content = response["choices"][0]["message"]["content"]

        # Optional: parse and validate the returned JSON
        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
