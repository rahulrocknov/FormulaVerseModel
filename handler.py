# handler.py
from google import genai
import os


# ✅ Initialize Gemini client
client = genai.Client(api_key="AIzaSyAHki864Y-BmDkHGgkI-rhoOkspXZ_q6QA")

def generate_output(equation: str) -> str:
    """
    Generate a short, realistic math word problem from a given equation using Gemini.
    """
    prompt = f"""
    You are a creative math teacher.
    Generate a short, realistic, and natural-sounding math word problem 
    that corresponds to this equation: {equation}.
    Keep it concise and ensure the numbers or operations make sense in a real-world scenario.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        return f"Error generating word problem: {str(e)}"
