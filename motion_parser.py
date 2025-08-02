import re
import spacy

nlp = spacy.load("en_core_web_sm")

def parse_problem(text):
    result = {
        "object": "projectile",
        "initial_velocity": None,
        "angle": None,
        "height": 0,
        "launch_type": "ground"
    }

    doc = nlp(text.lower())

    v_match = re.search(r'(?:initial velocity|speed) of (\d+)', text)
    if v_match:
        result["initial_velocity"] = float(v_match.group(1))

    angle_match = re.search(r'angle of (\d+)', text)
    if angle_match:
        result["angle"] = float(angle_match.group(1))

    h_match = re.search(r'from a height of (\d+)', text)
    if h_match:
        result["height"] = float(h_match.group(1))
        result["launch_type"] = "from_height"

    return result
