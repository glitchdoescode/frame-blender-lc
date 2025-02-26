# main.py
import random
import json
import os
import uuid
from typing import List, Dict, Optional, AsyncIterator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from langchain_service import LangChainService
from prompts import Prompts
from frame_hierarchy_analyzer import analyze_hierarchy, frame_relations, get_all_frames, get_related_frames
from utils import logs, log_message  # Import logs and log_message from utils.py

# Constants
FRAME_JSON_DIR = "frame_json"
EVALUATION_FILE = "data/evaluation.json"
VECTOR_STORE_PATH = "faiss_vector_store"
HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = "INFO"

# Options for dropdowns
PROMPTING_STRATEGIES = ["zero-shot", "one-shot", "few-shot", "chain-of-thought"]
RHETORICAL_OPTIONS = ["rhetorical", "non-rhetorical"]
FRAME_SOURCES = ["default", "frame_json"]
HIERARCHY_RELATIONS = list(frame_relations.keys())  # ["Inheritance", "Perspective", "Usage", "Subframe"]

# State Definition
class BlendingState(BaseModel):
    frames: List[str] = Field(..., description="List of frames to blend")
    settings: List[str] = Field(..., description="Prompting strategy and rhetorical setting")
    blended_expression: Optional[str] = Field(None, description="Generated blended expression")
    validation_result: Optional[str] = Field(None, description="Validation result")
    evaluation: Optional[Dict] = Field(None, description="Evaluation scores and notes")

# Utility Functions
def extract_frames_from_directory(dir_path: str) -> List[str]:
    """Read each .json file in dir_path and extract the 'frame_name'."""
    frames = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.json'):
            full_path = os.path.join(dir_path, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    frame_name = data.get("frame_name")
                    if frame_name:
                        frames.append(frame_name)
                log_message(f"INFO - Loaded frame from {filename}")
            except (json.JSONDecodeError, OSError) as e:
                log_message(f"WARNING - Could not process {filename}: {e}")
    return frames

def get_all_frames() -> List[str]:
    """Get all frames from frame_json directory."""
    return extract_frames_from_directory(FRAME_JSON_DIR)

def get_frames(frame_source: str, randomize_frames: bool, min_frames: int, max_frames: int, specific_frames: str, custom_frames: List[str] = None) -> List[str]:
    """Determine frames based on user inputs."""
    if frame_source == "default":
        all_frames = [f for f in custom_frames if f.strip()] if custom_frames else []
        if not all_frames:
            log_message("WARNING - No custom frames provided for default source, using empty list.")
    else:
        all_frames = extract_frames_from_directory(FRAME_JSON_DIR)
        if not all_frames:
            log_message("WARNING - No frames extracted from directory, using empty list.")

    log_message(f"INFO - Available frames: {all_frames}")
    if not all_frames:
        return []  # Return empty list if no frames are available

    if randomize_frames:
        num_frames = random.randint(min_frames, max_frames)
        return random.sample(all_frames, min(num_frames, len(all_frames)))
    elif specific_frames:
        specific_list = [f.strip() for f in specific_frames.split(",") if f.strip()]
        return [f for f in specific_list if f in all_frames] or all_frames[:max_frames]
    else:
        return all_frames[:max_frames]

def format_hierarchy(hierarchy_str: str) -> str:
    """Format the hierarchy string to ensure UTF-8 characters are preserved and indentation is correct."""
    lines = hierarchy_str.split('\n')
    formatted_lines = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if i == 0:  # Root node
            formatted_lines.append(line)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                prefix, name = parts
                # Ensure UTF-8 characters and proper indentation
                depth = (len(prefix) + 1) // 4  # Approximate depth based on prefix length
                indent = "│   " * (depth - 1) if depth > 1 else ""
                is_last = i == len(lines) - 1 or not lines[i + 1].strip()
                connector = "└── " if is_last else "├── "
                formatted_lines.append(f"{indent}{connector}{name}")
            else:
                formatted_lines.append(line)
    return '\n'.join(formatted_lines)

def ensure_data_directory():
    """Ensure the data directory and evaluation.json exist."""
    os.makedirs(os.path.dirname(EVALUATION_FILE), exist_ok=True)
    if not os.path.exists(EVALUATION_FILE):
        with open(EVALUATION_FILE, "w") as f:
            json.dump([], f)
    log_message("INFO - Ensured data directory and evaluation.json exist.")

def load_evaluations() -> List[Dict]:
    """Load existing evaluations from evaluation.json."""
    ensure_data_directory()
    try:
        with open(EVALUATION_FILE, "r") as f:
            evaluations = json.load(f)
        log_message("INFO - Loaded evaluations from evaluation.json.")
        return evaluations
    except Exception as e:
        log_message(f"ERROR - Failed to load evaluations: {e}")
        return []

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
    try:
        with open(EVALUATION_FILE, "w") as f:
            json.dump(evaluations, f, indent=4)
        log_message(f"INFO - Saved evaluation with id {entry['id']} to evaluation.json.")
    except Exception as e:
        log_message(f"ERROR - Failed to save evaluation: {e}")

# Initialize Services
langchain_service = LangChainService(vector_store_path=VECTOR_STORE_PATH)
load_result = langchain_service.load_packages()
if "Failed" in load_result:
    log_message(f"ERROR - Failed to initialize vector store: {load_result}")
    raise RuntimeError(f"Failed to initialize vector store: {load_result}")
log_message(f"INFO - Vector store initialization result: {load_result}")
llm = ChatOllama(model="llama3.2", temperature=1)
prompts = Prompts()

# Prompt Templates
def get_blending_instructions(settings: List[str]) -> str:
    """Get blending instructions based on settings."""
    prompting_strategy = settings[0].replace("-", "_")
    rhetorical = settings[1] == "rhetorical"
    method = getattr(prompts, prompting_strategy, None)
    if method is None:
        log_message(f"ERROR - Invalid prompting strategy: {prompting_strategy}")
        raise ValueError(f"Invalid prompting strategy: {prompting_strategy}")
    instructions = method(rhetorical)
    if not isinstance(instructions, str):
        log_message(f"ERROR - Instructions must be a string, got {type(instructions)}")
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

# LangGraph Nodes (Updated for async and checkpointers)
async def blending_node(state: BlendingState, config: Dict) -> Dict:
    log_message(f"INFO - Starting blending process for instance {config['configurable']['thread_id']} with frames: " + ", ".join(state.frames))
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
    
    log_message(f"INFO - Frames: {frames_str} | Context: {context[:100]}...")  # Truncate context for brevity
    if not retrieved_docs:
        log_message(f"WARNING - No documents retrieved for frames: {frames_str}")
    
    response = await langchain_service.generate_response_async(rag_chain, {"context": context, "input": frames_str, "instructions": instructions})  # Assume async method
    if not isinstance(response, str):
        log_message(f"ERROR - Blending response must be a string, got {type(response)}")
        raise ValueError(f"Blending response must be a string, got {type(response)}")
    state.blended_expression = response
    log_message("INFO - Blending completed successfully for instance " + config['configurable']['thread_id'] + ".")
    return state

async def validation_node(state: BlendingState, config: Dict) -> Dict:
    log_message(f"INFO - Starting validation process for instance {config['configurable']['thread_id']}.")
    if not state.blended_expression:
        state.validation_result = "Invalid: No blended expression provided"
        log_message(f"WARNING - No blended expression provided for validation in instance {config['configurable']['thread_id']}.")
        return state
    prompt = validation_prompt.format(
        frames=", ".join(state.frames),
        blended_expression=state.blended_expression
    )
    response = await llm.ainvoke(prompt)  # Use async invoke if available
    state.validation_result = response.content
    log_message(f"INFO - Validation completed for instance {config['configurable']['thread_id']}: " + state.validation_result)
    return state

async def evaluate_node(state: BlendingState, config: Dict) -> Dict:
    log_message(f"INFO - Starting evaluation process for instance {config['configurable']['thread_id']}.")
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
        log_message(f"WARNING - No blended expression provided for evaluation in instance {config['configurable']['thread_id']}.")
        return state
    prompt = evaluation_prompt.format(
        frames=", ".join(state.frames),
        blended_expression=state.blended_expression
    )
    response = await llm.ainvoke(prompt)  # Use async invoke if available
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
        log_message(f"INFO - Evaluation completed for instance {config['configurable']['thread_id']}: " + json.dumps(state.evaluation))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log_message(f"ERROR - Failed to parse evaluation response for instance {config['configurable']['thread_id']}: {e}, Cleaned Response: {cleaned_response}")
        state.evaluation = {
            "completeness": 0,
            "clarity": 0,
            "relevance": 0,
            "depth_of_understanding": 0,
            "coherence": 0,
            "execute_time": 0,
            "additional_notes": f"Failed to parse evaluation: {e}"
        }
    return state

async def storage_node(state: BlendingState, config: Dict) -> Dict:
    log_message(f"INFO - Starting storage process for instance {config['configurable']['thread_id']}.")
    entry = {
        "frames": state.frames,
        "settings": state.settings,
        "blended_expression": state.blended_expression,
        "evaluations": [state.evaluation],
    }
    save_evaluation(entry)
    log_message(f"INFO - Stored evaluation with id {entry['id']} for instance {config['configurable']['thread_id']}.")
    return state

# Set Up the Graph with Persistence
graph = StateGraph(BlendingState)
graph.add_node("blending", blending_node)
graph.add_node("validation", validation_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("storage", storage_node)
graph.add_edge(START, "blending")
graph.add_edge("blending", "validation")
graph.add_edge("validation", "evaluate")
graph.add_edge("evaluate", "storage")
graph.add_edge("storage", END)

# Use MemorySaver for checkpointer
checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

# FastAPI Frontend
app = FastAPI(title="Frame Blending API")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the HTML page with configuration options from index.html."""
    with open("index.html", "r") as f:
        html_content = f.read()

    # Populate dropdown options dynamically, setting "default" as selected
    frame_source_options = "".join(
        f'<option value="{source}" {"selected" if source == "default" else ""}>{source.capitalize()}</option>'
        for source in FRAME_SOURCES
    )
    prompting_strategy_options = "".join(f'<option value="{strategy}">{strategy}</option>' for strategy in PROMPTING_STRATEGIES)
    rhetorical_options = "".join(f'<option value="{option}">{option}</option>' for option in RHETORICAL_OPTIONS)
    hierarchy_relation_options = "".join(f'<option value="{relation}">{relation}</option>' for relation in HIERARCHY_RELATIONS)

    # Replace placeholders in the HTML
    html_content = html_content.replace("{{FRAME_SOURCE_OPTIONS}}", frame_source_options)
    html_content = html_content.replace("{{PROMPTING_STRATEGY_OPTIONS}}", prompting_strategy_options)
    html_content = html_content.replace("{{RHETORICAL_OPTIONS}}", rhetorical_options)
    html_content = html_content.replace("{{HIERARCHY_RELATION_OPTIONS}}", hierarchy_relation_options)

    return HTMLResponse(content=html_content)

@app.get("/logs", response_class=HTMLResponse)
async def get_logs(request: Request):
    """Serve the logs page displaying in-memory logs."""
    logs_html = "<h1>Application Logs</h1><div class='logs-container'>" + \
                "".join([f"<div class='log-entry'>{log}</div>" for log in logs]) + \
                "</div>"
    return HTMLResponse(content=logs_html)

@app.get("/output", response_class=HTMLResponse)
async def get_output(request: Request):
    """Serve the output history page displaying evaluation.json in reverse order."""
    evaluations = load_evaluations()
    evaluations.reverse()  # Reverse order (highest ID first)
    output_html = "<h1>Output History</h1><div class='output-container'>" + \
                  "".join([f"""
                    <div class='output-entry'>
                        <h3>Blend #{eval['id'] + 1}</h3>
                        <p><strong>Frames Used:</strong> {', '.join(eval['frames'])}</p>
                        <p><strong>Settings:</strong> {', '.join(eval['settings'])}</p>
                        <p><strong>Blended Expression:</strong> {eval['blending_result'] or 'Not generated'}</p>
                        <pre><strong>Evaluation:</strong> {json.dumps(eval['evaluations'][0], indent=2)}</pre>
                    </div>
                  """ for eval in evaluations]) + \
                  "</div>"
    return HTMLResponse(content=output_html)

@app.post("/hierarchy")
async def get_hierarchy(
    frame_source: str = Form(...),
    custom_frames: List[str] = Form(default=[]),
    hierarchy_relation: str = Form(...),
    hierarchy_direction: str = Form(...)
):
    """Generate and return the hierarchy tree for the active custom frame."""
    log_message("INFO - Hierarchy request received for frame source: " + frame_source)
    if frame_source != "default" or not custom_frames:
        log_message("WARNING - Invalid hierarchy request: frame source or custom frames missing.")
        return {"hierarchy": None}

    # Use only the first frame (active field value) for hierarchy generation
    active_frame = custom_frames[0].strip() if custom_frames and custom_frames[0].strip() else " "
    if not active_frame:
        log_message("WARNING - No active frame provided for hierarchy.")
        return {"hierarchy": None}

    reverse_order = hierarchy_direction == "parents"
    try:
        # Get all frames from frame_json to include related frames
        all_frames = get_all_frames()
        combined_frames = {active_frame}  # Start with the active frame

        # Recursively expand to include all related frames
        related_frames = get_related_frames(active_frame, hierarchy_relation, reverse_order, all_frames)
        combined_frames.update(related_frames)

        root = analyze_hierarchy(list(combined_frames), hierarchy_relation, reverse_order=reverse_order)
        hierarchy_str = format_hierarchy(str(root))
        log_message("INFO - Hierarchy generated for frame: " + active_frame)
        return {"hierarchy": hierarchy_str}
    except Exception as e:
        log_message(f"ERROR - Failed to generate hierarchy for frame {active_frame}: {str(e)}")
        return {"hierarchy": f"Error: {str(e)}"}

@app.post("/generate")
async def generate_blends(
    frame_source: str = Form(...),
    randomize_frames: str = Form(...),
    min_frames: int = Form(default=2),
    max_frames: int = Form(default=5),
    num_iterations: int = Form(default=1),
    specific_frames: str = Form(default=""),
    custom_frames: List[str] = Form(default=[]),
    prompting_strategy: str = Form(...),
    rhetorical: str = Form(...)
) -> JSONResponse:
    """Generate blends based on user-submitted form data."""
    instance_id = str(uuid.uuid4())  # Generate unique instance_id for each request
    log_message(f"INFO - Generate blends request received for instance {instance_id}.")
    randomize = randomize_frames.lower() == "true"
    settings = [prompting_strategy, rhetorical]
    frames = get_frames(frame_source, randomize, min_frames, max_frames, specific_frames, custom_frames)
    iterations = num_iterations if randomize else 1

    if not frames and randomize:
        log_message(f"WARNING - No frames provided for random blending in instance {instance_id}.")
        return JSONResponse({"results": [{"frames": [], "blended_expression": "No frames provided", "validation": "Invalid", "evaluation": {"additional_notes": "No frames to blend"}}]})
    elif not frames:
        log_message(f"WARNING - No frames provided for blending in instance {instance_id}.")
        return JSONResponse({"results": [{"frames": [], "blended_expression": "No frames provided", "validation": "Invalid", "evaluation": {"additional_notes": "No frames to blend"}}]})

    results = []
    config = {"configurable": {"thread_id": instance_id}}  # Use instance_id as thread_id for persistence

    for i in range(iterations):  # Changed from async for to regular for
        log_message(f"INFO - Generating blend iteration {i + 1} for instance {instance_id} with frames: {frames}")
        if randomize:  # Re-select frames for each iteration if randomizing
            frames = get_frames(frame_source, randomize, min_frames, max_frames, specific_frames, custom_frames)
        initial_state = BlendingState(frames=frames, settings=settings)
        final_state = await compiled_graph.ainvoke(initial_state, config=config)  # Use async invoke
        
        state_dict = final_state.get("values", {})
        result = {
            "frames": state_dict.get("frames", frames),
            "settings": state_dict.get("settings", settings),
            "blended_expression": state_dict.get("blended_expression"),
            "validation": state_dict.get("validation_result"),
            "evaluation": state_dict.get("evaluation")
        }
        results.append(result)

    log_message(f"INFO - Blending completed with {len(results)} results for instance {instance_id}.")
    return JSONResponse({"results": results, "instance_id": instance_id})