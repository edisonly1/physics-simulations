# text_to_sim.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load your API key from .env or environment variable
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_physics_info(prompt_text):
    system_prompt = """You are a physics assistant for AP Physics 1. From a problem like the one below, extract the setup as JSON with these fields:
{
  "objects": [{"name": "box", "mass": 2}],
  "forces": [{"type": "push", "magnitude": 10, "direction": "right"}],
  "friction": false,
  "motion_type": "linear",
  "question": "acceleration"
}
Problem examples might involve projectiles, inclines, or forces on objects."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {prompt_text}"}
        ]
    )

    return response.choices[0].message.content
