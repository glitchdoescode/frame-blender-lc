# config.py

# General settings
FRAME_JSON_DIR = "frame_json"  # Directory containing frame JSON files
EVALUATION_FILE = "data/evaluation.json"  # File to store evaluation results
VECTOR_STORE_PATH = "faiss_vector_store"  # Path to save/load FAISS vector store

# Frame selection configuration
USE_DEFAULT_FRAMES = False  # If True, uses DEFAULT_FRAMES; if False, extracts from FRAME_JSON_DIR
DEFAULT_FRAMES = [
    "Travel", "Aging", "Time", "Money", "Health",
    "Education", "Technology", "Environment", "Politics", "Culture"
]  # Fallback frames if USE_DEFAULT_FRAMES is True
MIN_FRAMES = 2  # Minimum number of frames to select if random
MAX_FRAMES = 5  # Maximum number of frames to select if random

# Generation behavior
RANDOMIZE_FRAMES = True  # If True, randomly selects frames; if False, uses all or specified frames
NUM_ITERATIONS = 5  # Number of blending iterations if RANDOMIZE_FRAMES is True; set to 1 for single run
SPECIFIC_FRAMES = []  # List of specific frames to use if RANDOMIZE_FRAMES is False (e.g., ["Time", "Money"])

# Prompting and rhetorical settings
PROMPTING_STRATEGY = "chain-of-thought"  # Options: "zero-shot", "one-shot", "few-shot", "chain-of-thought"
RHETORICAL = "rhetorical"  # Options: "rhetorical", "non-rhetorical"

# FastAPI settings
HOST = "0.0.0.0"  # Host for FastAPI server
PORT = 8000  # Port for FastAPI server

# Logging settings
LOG_FILE = "blending_context.log"
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"