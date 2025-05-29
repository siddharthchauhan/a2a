import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")
warnings.filterwarnings("ignore", category=UserWarning, module="crewai.telemtry.telemetry")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from fastapi import FastAPI
from a2a_utils import to_envelope
import asyncio
from concurrent.futures import ThreadPoolExecutor
from crewai import Agent, Task, Crew

app = FastAPI(title="AI-Image-Writer")

# Create a thread pool executor for running sync code
executor = ThreadPoolExecutor()

# Define the CrewAI agent and task
writer_agent = Agent(
    role="Poetic copywriter",
    goal="Write witty one-sentence image descriptions",
    backstory=(
        "A creative writer employed by a design studio to craft short "
        "captions that make people smile."
    ),
    allow_delegation=False,
)

caption_task = Task(
    description=(
        "Create a witty one-sentence description for the image prompt: {prompt}"
    ),
    expected_output="One witty sentence describing the image",
    agent=writer_agent,
)

def run_crew(prompt: str) -> str:
    """Execute the CrewAI task to generate a witty caption."""
    crew = Crew(
        agents=[writer_agent],
        tasks=[caption_task],
        verbose=False,
    )
    # kickoff is synchronous so we run it in a thread executor in the endpoint
    return crew.kickoff(inputs={"prompt": prompt})

# 2️⃣  Expose A2A endpoint
@app.post("/a2a")
async def a2a_call(request: dict):
    # Access task directly without checking method
    task_data = request["params"]["task"]
    prompt = task_data["input"]["text"]

    # Run the CrewAI logic in a thread pool to avoid blocking
    response_text = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: run_crew(prompt)
    )
    
    response_env = to_envelope(
        "a2a.status",
        {
            "task_id": task_data["id"],
            "state": "completed",
            "output": {"text": response_text}
        }
    )
    return response_env


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
