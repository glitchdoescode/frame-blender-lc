#main.py
import random
import json
import os
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from langchain_service import LangChainService
from prompts import Prompts
import logging

# Configure logging
logging.basicConfig(
    filename="blending_context.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
EVALUATION_FILE = "data/evaluation.json"
ALL_FRAMES = [
    "Travel", "Aging", "Time", "Money", "Health",
    "Education", "Technology", "Environment", "Politics", "Culture"
]
PROMPTING_STRATEGIES = ["zero-shot", "one-shot", "few-shot", "chain-of-thought"]
RHETORICAL_OPTIONS = ["rhetorical", "non-rhetorical"]

# State Definition
class BlendingState(BaseModel):
    frames: List[str] = Field(..., description="List of frames to blend")
    settings: List[str] = Field(..., description="Prompting strategy and rhetorical setting")
    blended_expression: Optional[str] = Field(None, description="Generated blended expression")
    validation_result: Optional[str] = Field(None, description="Validation result")
    evaluation: Optional[Dict] = Field(None, description="Evaluation scores and notes")

# Utility Functions
def select_random_frames(min_frames=2, max_frames=5) -> List[str]:
    """Select a random number of frames from ALL_FRAMES."""
    num_frames = random.randint(min_frames, max_frames)
    return random.sample(ALL_FRAMES, num_frames)

def select_random_settings() -> List[str]:
    """Select random prompting strategy and rhetorical setting."""
    prompting = random.choice(PROMPTING_STRATEGIES)
    rhetorical = random.choice(RHETORICAL_OPTIONS)
    return [prompting, rhetorical]

def ensure_data_directory():
    """Ensure the data directory and evaluation.json exist."""
    os.makedirs(os.path.dirname(EVALUATION_FILE), exist_ok=True)
    if not os.path.exists(EVALUATION_FILE):
        with open(EVALUATION_FILE, "w") as f:
            json.dump([], f)

def load_evaluations() -> List[Dict]:
    """Load existing evaluations from evaluation.json."""
    ensure_data_directory()
    with open(EVALUATION_FILE, "r") as f:
        return json.load(f)

def save_evaluation(entry: Dict):
    """Append a new evaluation entry to evaluation.json."""
    evaluations = load_evaluations()
    entry["id"] = len(evaluations)
    for key, value in entry.items():
        try:
            json.dumps(value)
        except TypeError:
            raise ValueError(f"Non-serializable value in {key}: {value}")
    evaluations.append(entry)
    with open(EVALUATION_FILE, "w") as f:
        json.dump(evaluations, f, indent=4)

# Initialize Services
langchain_service = LangChainService(vector_store_path="faiss_vector_store")
load_result = langchain_service.load_packages()
if "Failed" in load_result:
    raise RuntimeError(f"Failed to initialize vector store: {load_result}")
logging.info(f"Vector store initialization result: {load_result}")
llm = ChatOllama(model="llama3.2", temperature=1)
prompts = Prompts()

# Prompt Templates
def get_blending_instructions(settings: List[str]) -> str:
    """Get blending instructions based on settings."""
    prompting_strategy = settings[0].replace("-", "_")
    rhetorical = settings[1] == "rhetorical"
    method = getattr(prompts, prompting_strategy, None)
    if method is None:
        raise ValueError(f"Invalid prompting strategy: {prompting_strategy}")
    instructions = method(rhetorical)
    if not isinstance(instructions, str):
        raise ValueError(f"Instructions must be a string, got {type(instructions)}")
    return instructions

blending_prompt_template = """
{instructions}

Below is some context about frames (extracted from the user’s frame collection):
{context}

The user wants to blend the following frames: {input}

Please produce:
1. A short example sentence or expression that illustrates how these frames blend.
2. A concise analysis explaining the input spaces, cross-space mapping, blended space, and emergent structure.

If you are missing crucial info or are unsure, respond with "I don't know."

Answer concisely, in a professional style.
"""

validation_prompt = PromptTemplate(
    input_variables=["frames", "blended_expression"],
    template="""
You are a validation agent. Your task is to check if the following blended expression correctly and coherently combines the given frames.

Frames: {frames}
Blended Expression: {blended_expression}

Respond with "Valid" if the blend is coherent and meaningful, or "Invalid" with a brief explanation if it's not.
"""
)

evaluation_prompt = PromptTemplate(
    input_variables=["frames", "blended_expression"],
    template="""
You are an evaluation agent. Your task is to evaluate the following blended expression based on the given frames.

Frames: {frames}
Blended Expression: {blended_expression}

Evaluate the blend on the following criteria, each on a scale of 0 to 10:
- completeness: Does the blend fully incorporate all input frames?
- clarity: Is the blend easy to understand?
- relevance: Are the frames blended in a way that makes sense together?
- depth_of_understanding: Does the blend show a deep understanding of the frames?
- coherence: Is the blend logically consistent?
- execute_time: Is the blend creative and well-executed?

Provide a JSON object with the scores and justifications, ensuring the response is a single-line string with escaped newlines (\\n) if needed, like this:
{{
    "completeness": 8,
    "clarity": 9,
    "relevance": 7,
    "depth_of_understanding": 8,
    "coherence": 9,
    "execute_time": 6,
    "justifications": "Brief explanation with no unescaped newlines"
}}
"""
)

# LangGraph Nodes
def blending_node(state: BlendingState) -> BlendingState:
    instructions = get_blending_instructions(state.settings)
    prompt_template = PromptTemplate(
        template=blending_prompt_template,
        input_variables=["context", "input", "instructions"]
    )
    rag_chain = langchain_service.build_rag_chain(prompt_template)
    frames_str = ", ".join(state.frames)
    
    retriever = langchain_service.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(frames_str)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."
    
    logging.info(f"Frames: {frames_str} | Context: {context}")
    if not retrieved_docs:
        logging.warning(f"No documents retrieved for frames: {frames_str}")
    
    response = langchain_service.generate_response(rag_chain, {"context": context, "input": frames_str, "instructions": instructions})
    if not isinstance(response, str):
        raise ValueError(f"Blending response must be a string, got {type(response)}")
    state.blended_expression = response
    return state

def validation_node(state: BlendingState) -> BlendingState:
    """Validate the blended expression."""
    if not state.blended_expression:
        state.validation_result = "Invalid: No blended expression provided"
        return state
    prompt = validation_prompt.format(
        frames=", ".join(state.frames),
        blended_expression=state.blended_expression
    )
    response = llm.invoke(prompt)
    state.validation_result = response.content
    return state

def evaluate_node(state: BlendingState) -> BlendingState:
    """Evaluate the blended expression."""
    if not state.blended_expression:
        state.evaluation = {
            "completeness": 0,
            "clarity": 0,
            "relevance": 0,
            "depth_of_understanding": 0,
            "coherence": 0,
            "execute_time": 0,
            "additional_notes": "No blended expression to evaluate"
        }
        return state
    prompt = evaluation_prompt.format(
        frames=", ".join(state.frames),
        blended_expression=state.blended_expression
    )
    response = llm.invoke(prompt)
    print(f"Raw Evaluation Response: {repr(response.content)}")
    cleaned_response = ''.join(response.content.splitlines()).strip()
    try:
        eval_data = json.loads(cleaned_response)
        state.evaluation = {
            "completeness": int(eval_data["completeness"]),
            "clarity": int(eval_data["clarity"]),
            "relevance": int(eval_data["relevance"]),
            "depth_of_understanding": int(eval_data["depth_of_understanding"]),
            "coherence": int(eval_data["coherence"]),
            "execute_time": int(eval_data["execute_time"]),
            "additional_notes": eval_data.get("justifications", "")
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Failed to parse evaluation response: {e}, Cleaned Response: {cleaned_response}")
        state.evaluation = {
            "completeness": 0,
            "clarity": 0,
            "relevance": 0,
            "depth_of_understanding": 0,
            "coherence": 0,
            "execute_time": 0,
            "additional_notes": f"Failed to parse evaluation: {str(e)}"
        }
    return state

def storage_node(state: BlendingState) -> BlendingState:
    """Store the results in evaluation.json."""
    entry = {
        "frames": state.frames,
        "settings": state.settings,
        "blending_result": state.blended_expression,
        "evaluations": [state.evaluation],
    }
    save_evaluation(entry)
    return state

# Set Up the Graph
graph = StateGraph(BlendingState)
graph.add_node("blending", blending_node)
graph.add_node("validation", validation_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("storage", storage_node)
graph.add_edge("blending", "validation")
graph.add_edge("validation", "evaluate")
graph.add_edge("evaluate", "storage")
graph.add_edge("storage", END)
graph.set_entry_point("blending")
compiled_graph = graph.compile()

# Main Loop
if __name__ == "__main__":
    for _ in range(5):  # Run 5 times with random frames and settings
        frames = select_random_frames()
        settings = select_random_settings()
        initial_state = BlendingState(frames=frames, settings=settings)
        final_state = compiled_graph.invoke(initial_state)
        print(f"Processed frames: {final_state['frames']}")
        print(f"Settings: {final_state['settings']}")
        print(f"Blended Expression: {final_state['blended_expression']}")
        print(f"Validation: {final_state['validation_result']}")
        print(f"Evaluation: {final_state['evaluation']}")
        print("-" * 50)