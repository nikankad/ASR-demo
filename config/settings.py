"""Configuration settings for the ASR application."""

import os

# Model configuration
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "models/model_weights/best.pt")
LM_MODEL_NAME = os.getenv("LM_MODEL_NAME", "models/model_weights/3-gram.pruned.3e-7.arpa")  # HF model name or local path
LM_MODEL_PATH = os.getenv("LM_MODEL_PATH", "models/model_weights/3-gram.pruned.3e-7.arpa")
SAMPLE_RATE = 16000
AUDIO_MAX_DURATION = 30  # seconds

# Gradio UI configuration
UI_TITLE = "IBNet Demo"
UI_DESCRIPTION = "Automatic Speech Recognition"
THEME = "dark"
