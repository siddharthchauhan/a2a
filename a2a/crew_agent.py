import warnings
import importlib

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._config",
)

if importlib.util.find_spec("crewai.telemetry.telemetry") is not None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="crewai.telemetry.telemetry",
    )

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from a2a_utils import to_envelope
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

app = FastAPI(title="AI-Image-Writer")

# Create a thread pool executor for running sync code
executor = ThreadPoolExecutor()

# Initialize OpenAI client
client = OpenAI()

# 2️⃣  Expose A2A endpoint
@app.post("/a2a")
async def a2a_call(request: dict):
    # Access task directly without checking method
    task_data = request["params"]["task"]
    prompt = task_data["input"]["text"]

    # Create system prompt for the creative task
    system_prompt = """You are a poetic copywriter hired by a design studio.
    Your goal is to describe any image prompt in one witty sentence.
    Make it creative, clever and concise."""
    
    # Run the AI request in a thread pool to avoid blocking
    result = await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",  # Or another available model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a witty one-sentence description for: {prompt}"}
            ],
            temperature=0.7,
            max_tokens=100
        )
    )
    
    # Extract the response text
    response_text = result.choices[0].message.content.strip()
    
    response_env = to_envelope(
        "a2a.status",
        {
            "task_id": task_data["id"],
            "state": "completed",
            "output": {"text": response_text}
        }
    )
    return response_env
