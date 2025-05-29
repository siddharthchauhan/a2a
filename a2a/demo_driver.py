import httpx, asyncio, uuid, json, argparse
from a2a_utils import to_envelope
import re
from openai import AsyncOpenAI

# Initialize OpenAI client for the supervisor agent
openai_client = AsyncOpenAI()

async def call_agent(client, url, task_name, question, agent_name):
    """Make a call to an A2A agent and print the response."""
    task_id = str(uuid.uuid4())
    envelope = to_envelope(
        "a2a.call",
        {
            "task": {
                "id": task_id,
                "name": task_name,
                "input": {"text": question}
            }
        }
    )
    
    print(f"\nSending request to {agent_name} at {url}...")
    print(f"Question: \"{question}\"")
    
    try:
        res = await client.post(url, json=envelope, timeout=30.0)
        result = res.json()
        print(f"{agent_name} replied:", json.dumps(result, indent=2))
        
        # Extract the actual text response from the result
        if "params" in result and "output" in result["params"] and "text" in result["params"]["output"]:
            return result["params"]["output"]["text"]
        return None
    except Exception as e:
        print(f"Error calling {agent_name}: {str(e)}")
        return None

async def supervisor_agent(query, crew_description, lang_description):
    """A supervisor agent that decides if and how collaboration should happen."""
    print("\n🧠 SUPERVISOR AGENT: Analyzing your query...")
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a supervisor agent that decides how to route user queries between two specialized AI agents:
1. CrewAI - Specializes in creative writing, humor, storytelling, and engaging content.
2. LangGraph - Specializes in technical explanations, math, science, and factual information.

You have three options:
1. Route to CrewAI only
2. Route to LangGraph only
3. Collaborate (route to one first, then pass its output to the other)

For collaboration, also decide which agent should go first.
Format your response exactly as follows:
DECISION: [SINGLE or COLLABORATE]
ROUTE: [CREW_ONLY or LANG_ONLY or CREW_FIRST or LANG_FIRST]
REASON: [Brief explanation of your decision]"""},
                {"role": "user", "content": f"""Query: "{query}"

CrewAI Description: {crew_description}
LangGraph Description: {lang_description}

Analyze this query and decide how to route it."""}
            ],
            temperature=0.2,
            max_tokens=150
        )
        
        decision_text = response.choices[0].message.content.strip()
        print(f"Supervisor decision: {decision_text}")
        
        # Extract decision with more flexible pattern matching
        # First try to get all components with regex
        decision = None
        route = None
        reason = None
        
        # Try to extract decision with various patterns
        decision_patterns = [
            r'DECISION:\s*(SINGLE|COLLABORATE|LANG_ONLY|CREW_ONLY)',
            r'DECISION\s*:\s*(SINGLE|COLLABORATE|LANG_ONLY|CREW_ONLY)'
        ]
        
        for pattern in decision_patterns:
            match = re.search(pattern, decision_text, re.IGNORECASE)
            if match:
                decision = match.group(1).upper()
                if decision in ["LANG_ONLY", "CREW_ONLY"]:
                    route = decision
                    decision = "SINGLE"
                break
        
        # Try to extract route with various patterns if not already set
        if not route:
            route_patterns = [
                r'ROUTE:\s*(CREW_ONLY|LANG_ONLY|CREW_FIRST|LANG_FIRST)',
                r'ROUTE\s*:\s*(CREW_ONLY|LANG_ONLY|CREW_FIRST|LANG_FIRST)'
            ]
            
            for pattern in route_patterns:
                match = re.search(pattern, decision_text, re.IGNORECASE)
                if match:
                    route = match.group(1).upper()
                    break
        
        # Try to extract reason
        reason_pattern = r'REASON:\s*(.*?)(?:\n|$)'
        reason_match = re.search(reason_pattern, decision_text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"
        
        # Fallback logic for decision/routing
        if not decision:
            # Try to infer from content if formats don't match
            if "technical" in decision_text.lower() or "factual" in decision_text.lower():
                decision = "SINGLE"
                route = "LANG_ONLY"
            elif "creative" in decision_text.lower() or "storytelling" in decision_text.lower():
                decision = "SINGLE"
                route = "CREW_ONLY"
            elif "collaborate" in decision_text.lower():
                decision = "COLLABORATE"
                route = "LANG_FIRST" if "lang first" in decision_text.lower() else "CREW_FIRST"
            else:
                # Default case
                decision = "SINGLE"
                route = "LANG_ONLY"
        
        # If we have a decision but no route, set a default route
        if decision and not route:
            if decision == "SINGLE":
                route = "LANG_ONLY"  # Default to LangGraph for single agent mode
            elif decision == "COLLABORATE":
                route = "LANG_FIRST"  # Default to LangGraph first for collaboration
        
        # Display the parsed decision
        if decision == "COLLABORATE":
            print(f"\n🔄 COLLABORATION RECOMMENDED 🔄")
            print(f"Routing: {'CrewAI → LangGraph' if route == 'CREW_FIRST' else 'LangGraph → CrewAI'}")
            print(f"Reason: {reason}")
            return decision, route, reason
        else:
            print(f"\n🔀 SINGLE AGENT MODE 🔀")
            print(f"Routing to: {'CrewAI' if route == 'CREW_ONLY' else 'LangGraph'}")
            print(f"Reason: {reason}")
            return decision, route, reason
            
    except Exception as e:
        print(f"\n⚠️ Error with supervisor agent: {str(e)}. Defaulting to LangGraph only.")
        return "SINGLE", "LANG_ONLY", "Error fallback"

async def main():
    # Set up argument parser for server configuration
    parser = argparse.ArgumentParser(description="A2A Demo Driver - With Supervisor Agent")
    parser.add_argument("--crew-url", default="http://localhost:8001/a2a", help="URL for the CrewAI agent")
    parser.add_argument("--lang-url", default="http://localhost:8002/a2a", help="URL for the LangGraph agent")
    parser.add_argument("--task-name", default="caption-an-image", help="Task name to use in the A2A protocol")
    
    args = parser.parse_args()
    
    # Agent descriptions for the supervisor
    crew_description = "Creative writing agent that generates engaging, witty content. Good for storytelling, creative descriptions, and humorous takes."
    lang_description = "Technical explanation agent that provides clear, factual information. Good for math, science, history, and detailed explanations."
    
    # Interactive user prompts
    print("A2A Demo Driver - Supervised Collaboration Mode")
    print("=============================================")
    print("The supervisor agent will decide how to route your query")
    
    user_prompt = input("\nEnter your question or prompt:\n> ")
    if not user_prompt.strip():
        user_prompt = "Explain the Pythagorean theorem"
        print(f"Using default prompt: \"{user_prompt}\"")
    
    # Get decision from supervisor agent
    decision, route, reason = await supervisor_agent(user_prompt, crew_description, lang_description)
    
    async with httpx.AsyncClient() as client:
        if decision == "COLLABORATE":
            if route == "CREW_FIRST":
                # CrewAI → LangGraph
                crew_response = await call_agent(client, args.crew_url, args.task_name, user_prompt, "CrewAI")
                
                if crew_response:
                    print(f"\n🔄 Passing CrewAI's response to LangGraph...")
                    await call_agent(client, args.lang_url, args.task_name, crew_response, "LangGraph")
            else:  # LANG_FIRST
                # LangGraph → CrewAI
                lang_response = await call_agent(client, args.lang_url, args.task_name, user_prompt, "LangGraph")
                
                if lang_response:
                    print(f"\n🔄 Passing LangGraph's response to CrewAI...")
                    await call_agent(client, args.crew_url, args.task_name, lang_response, "CrewAI")
        else:  # SINGLE
            if route == "CREW_ONLY":
                # Just use CrewAI
                await call_agent(client, args.crew_url, args.task_name, user_prompt, "CrewAI")
            else:  # LANG_ONLY
                # Just use LangGraph
                await call_agent(client, args.lang_url, args.task_name, user_prompt, "LangGraph")

if __name__ == "__main__":
    asyncio.run(main())
