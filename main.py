# --- Imports ---
from langgraph.graph import StateGraph, Node
from langchain_community.llms import ChatGroq
from fastapi import FastAPI
import logging
import os

logging.basicConfig(level=logging.INFO)



app = FastAPI(title="CodeGen Agent")



# --- Set up Groq API Key (temporary quick test only) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Define Nodes ---

def context_fetcher(state):
    """Fetches and structures the user’s coding topic."""
    context = state.get("context", None)
    if not context:
        raise ValueError("Missing context. Provide 'context' key in state.")
    state['topic'] = f"Generate Python code for: {context}"
    return state


def code_writer(state):
    """Writes complete, efficient, and well-documented code using Groq."""
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    query = f"Write clean, efficient, and well-commented Python code for: {state['topic']}."
    resp = llm.invoke(query)
    state['raw_code'] = resp  # ChatGroq.invoke() already returns text
    return state


def code_refiner(state):
    """Refines and optimizes the generated code."""
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    query = f"Refactor and optimize the following code for clarity and performance:\n\n{state['raw_code']}"
    resp = llm.invoke(query)
    state['final_code'] = resp
    return state


def evaluate_quality(state):
    """Evaluates code quality (mock quality check)."""
    code = state.get("final_code", "")
    if "import" in code or "def" in code:
        state['quality'] = "good"
    else:
        state['quality'] = "bad"
    return state


def format_final(state):
    """Formats the code output nicely."""
    formatted = f"# --- Final Generated Code ---\n\n{state['final_code']}\n"
    state['formatted_code'] = formatted
    return state


# --- Build Graph ---

graph = StateGraph()

# Add nodes
graph.add_node("context_fetcher", Node(context_fetcher))
graph.add_node("code_writer", Node(code_writer))
graph.add_node("code_refiner", Node(code_refiner))
graph.add_node("evaluate_quality", Node(evaluate_quality))
graph.add_node("format_final", Node(format_final))

# Define edges (flow)
graph.add_edge("context_fetcher", "code_writer")
graph.add_edge("code_writer", "code_refiner")
graph.add_edge("code_refiner", "evaluate_quality")

# Conditional branching
graph.add_edge("evaluate_quality", "format_final", condition=lambda s: s["quality"] == "good")
graph.add_edge("evaluate_quality", "code_refiner", condition=lambda s: s["quality"] == "bad")

graph.add_edge("format_final", "END")

# --- Compile Graph ---
compiled_graph = graph.compile()

# --- Run Graph ---
def run_agent(context: str) -> str:
    state = {"context": context}
    result = compiled_graph.invoke(state)
    return result["formatted_code"]

@app.post("/generate")
async def generate(payload: dict):
    context = payload.get("context")
    if not context:
        return {"error": "Missing context"}
    output = run_agent(context)
    return {"output": output}


