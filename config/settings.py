"""Configuration settings for the ASR application."""

import os


def _required_env(name):
    value = os.getenv(name)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _required_int(name):
    return int(_required_env(name))


def _required_float(name):
    return float(_required_env(name))

# Model configuration
MODEL_BACKEND = _required_env("MODEL_BACKEND")
MODEL_WEIGHTS_PATH = _required_env("MODEL_WEIGHTS_PATH")
VOCAB = _required_env("VOCAB")
MODEL_CLASS = os.getenv("MODEL_CLASS", "")
LM_MODEL_NAME = _required_env("LM_MODEL_NAME")  # HF model name or local path
LM_MODEL_PATH = _required_env("LM_MODEL_PATH")
LM_ALPHA = _required_float("LM_ALPHA")
LM_BETA = _required_float("LM_BETA")
LM_UNK_SCORE_OFFSET = _required_float("LM_UNK_SCORE_OFFSET")
SAMPLE_RATE = _required_int("SAMPLE_RATE")
BEAM_WIDTH = _required_int("BEAM_WIDTH")
AUDIO_MAX_DURATION = _required_int("AUDIO_MAX_DURATION")  # seconds

# Gradio UI configuration
UI_TITLE = _required_env("UI_TITLE")
UI_DESCRIPTION = _required_env("UI_DESCRIPTION")
THEME = _required_env("THEME")
