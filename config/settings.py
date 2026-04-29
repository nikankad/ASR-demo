"""Configuration settings for the ASR application."""

import os

# Model configuration
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "ibnet")
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "models/model_weights/best.pt")
VOCAB = os.getenv("VOCAB", "abcdefghijklmnopqrstuvwxyz '")
MODEL_CLASS = os.getenv("MODEL_CLASS", "")
LM_MODEL_NAME = os.getenv("LM_MODEL_NAME", "models/model_weights/6gram.bin")  # HF model name or local path
LM_MODEL_PATH = os.getenv("LM_MODEL_PATH", "models/model_weights/6gram.bin")
LM_ALPHA = float(os.getenv("LM_ALPHA", "0.15"))
LM_BETA = float(os.getenv("LM_BETA", "0.0"))
LM_UNK_SCORE_OFFSET = float(os.getenv("LM_UNK_SCORE_OFFSET", "-10.0"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
BEAM_WIDTH = int(os.getenv("BEAM_WIDTH", "100"))
AUDIO_MAX_DURATION = int(os.getenv("AUDIO_MAX_DURATION", "30"))  # seconds

# Gradio UI configuration
UI_TITLE = os.getenv("UI_TITLE", "IBNet Demo")
UI_DESCRIPTION = os.getenv("UI_DESCRIPTION", "Automatic Speech Recognition")
THEME = os.getenv("THEME", "dark")
