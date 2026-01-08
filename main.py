# --- Imports ---
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import os

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="CodeGen Agent")

# --- Pydantic model for request ---
class GenerateRequest(BaseModel):
    context: str

# --- Set up Groq API Key ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required.")

# --- Define State ---
class AgentState(TypedDict):
    context: str
    topic: str
    raw_code: str
    final_code: str
    quality: str
    formatted_code: str

# --- Define Nodes (return dict for partial state updates) ---

def context_fetcher(state: AgentState) -> Dict[str, Any]:
    """Fetches and structures the user's coding topic."""
    context = state.get("context", "")
    if not context:
        raise ValueError("Missing context. Provide 'context' key in state.")
    return {"topic": f"Generate Python code for: {context}"}

def code_writer(state: AgentState) -> Dict[str, Any]:
    """Writes complete, efficient, and well-documented code using Groq."""
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    query = f"Write clean, efficient, and well-commented Python code for: {state['topic']}."
    resp = llm.invoke([HumanMessage(content=query)]).content
    return {"raw_code": resp}

def code_refiner(state: AgentState) -> Dict[str, Any]:
    """Refines and optimizes the generated code."""
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    query = f"Refactor and optimize the following code for clarity and performance:\n\n{state['raw_code']}"
    resp = llm.invoke([HumanMessage(content=query)]).content
    return {"final_code": resp}

def evaluate_quality(state: AgentState) -> Dict[str, Any]:
    """Evaluates code quality (mock quality check)."""
    code = state.get("final_code", "")
    if "import" in code or "def" in code:
        return {"quality": "good"}
    else:
        return {"quality": "bad"}

def format_final(state: AgentState) -> Dict[str, Any]:
    """Formats the code output nicely."""
    formatted = f"# --- Final Generated Code ---\n\n{state['final_code']}\n"
    return {"formatted_code": formatted}

# --- Build Graph ---

workflow = StateGraph(AgentState)

# Add nodes - pass functions directly
workflow.add_node("context_fetcher", context_fetcher)
workflow.add_node("code_writer", code_writer)
workflow.add_node("code_refiner", code_refiner)
workflow.add_node("evaluate_quality", evaluate_quality)
workflow.add_node("format_final", format_final)

# Define edges
workflow.add_edge(START, "context_fetcher")
workflow.add_edge("context_fetcher", "code_writer")
workflow.add_edge("code_writer", "code_refiner")

# Conditional branching after evaluate_quality
def route_quality(state: AgentState) -> str:
    return "format_final" if state["quality"] == "good" else "code_refiner"

workflow.add_conditional_edges(
    "evaluate_quality",
    route_quality,
    {
        "format_final": "format_final",
        "code_refiner": "code_refiner"
    }
)

workflow.add_edge("format_final", END)

# Compile Graph
app_graph = workflow.compile()

# --- FastAPI Endpoint ---
@app.post("/generate")
async def generate(payload: GenerateRequest):
    initial_state = {"context": payload.context}
    result = app_graph.invoke(initial_state)
    return {"output": result["formatted_code"]}
