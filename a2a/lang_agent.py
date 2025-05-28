import warnings
import asyncio
from typing import Dict, Any

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from fastapi import FastAPI
from openai import AsyncOpenAI
from a2a_utils import to_envelope

# Initialize the OpenAI client directly
client = AsyncOpenAI()
app = FastAPI(title="LangGraph-Math-Explainer")

# Simple function to generate explanations without using LangGraph
async def generate_explanation(question: str) -> str:
    """Generate a simple explanation for a given question using OpenAI."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains math concepts in two lines."},
            {"role": "user", "content": f"Explain in two lines: {question}"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content

# Expose A2A endpoint
@app.post("/a2a")
async def a2a_call(request: dict):
    task = request["params"]["task"]
    question = task["input"]["text"]
    
    # Generate explanation
    explanation = await generate_explanation(question)
    
    return to_envelope(
        "a2a.status",
        {
            "task_id": task["id"], 
            "state": "completed", 
            "output": {"text": explanation}
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)