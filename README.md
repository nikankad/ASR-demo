# ASR Live Demo

A Gradio-based web interface for automatic speech recognition with real-time transcription. This project is built to be reusable: bring your own model weights, vocabulary, and optional language model.

## Features

- Upload audio files or record from microphone
- Dual decoding output: raw greedy CTC and optional beam/LM decoding
- Bring-your-own model backend support (`ibnet` or custom Python class)
- Config-driven setup with `.env` values
- Model information panel (backend, vocab size, parameters, LM status)
- Live resource charts (CPU, RAM, GPU memory)

## Requirements

- Python 3.11
- PyTorch + TorchAudio
- Gradio
- numpy
- librosa
- pyctcdecode
- pypi-kenlm (optional, only needed for external KenLM scoring)
- psutil, pandas (resource panel)

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1. Copy `.env.example` to `.env`.
2. Fill in model-specific values.
3. Run:

```bash
python app.py
```

App default URL: `http://localhost:7860`

## Configuration

All settings are environment-driven via `.env`:

- `MODEL_BACKEND`: `ibnet` or `python_class`
- `MODEL_WEIGHTS_PATH`: path to your acoustic model weights/checkpoint
- `VOCAB`: character/token vocabulary used for CTC decoding
- `MODEL_CLASS`: required when `MODEL_BACKEND=python_class`, format `package.module:ClassName`
- `LM_MODEL_NAME`, `LM_MODEL_PATH`: optional LM name/path for decoder
- `LM_ALPHA`, `LM_BETA`, `LM_UNK_SCORE_OFFSET`, `BEAM_WIDTH`: beam/LM controls
- `SAMPLE_RATE`, `AUDIO_MAX_DURATION`: audio settings
- `UI_TITLE`, `UI_DESCRIPTION`, `THEME`: UI settings

Note: `.env.example` is a template. Put real local paths only in your `.env`.

## Model Integration Options

### 1) Built-in IBNet backend

Set:

```bash
MODEL_BACKEND=ibnet
MODEL_WEIGHTS_PATH=path/to/your/model_checkpoint.pt
VOCAB=your_vocab_string
```

### 2) Custom Python backend

Use when your model architecture differs from IBNet.

Set:

```bash
MODEL_BACKEND=python_class
MODEL_CLASS=models.custom_backend_example:CustomASRBackend
MODEL_WEIGHTS_PATH=path/to/your/model_checkpoint.pt
VOCAB=your_vocab_string
```

Implement a class with:

- `__init__(self, model_path, device="cpu", vocab=None)`
- `transcribe(self, audio_waveform)`

Return a dict with at least:

```python
{"text": "..."}
```

Optional return keys:

- `confidence`: float in `[0, 1]`
- `logits`: if provided in CTC-compatible shape, LM/beam decoding can run on it

Starter template: `models/custom_backend_example.py`

## Optional Language Model (LM)

You can run without an LM. If configured, LM decoding is shown in the "With Language Model" panel.

- `LM_MODEL_PATH` can point to your LM artifact (for example `.arpa` or `.bin` as supported by your setup)
- If LM is unavailable, the app falls back to raw output in the LM panel

No specific LM filename is required by this repo.

## UI Notes

- Transcription outputs are shown side by side:
  - Raw (Greedy CTC)
  - With Language Model
- "Model Panel" includes:
  - Model Information tab
  - Resource Usage tab with live charts

## Project Structure

- `app.py`: Gradio app and inference flow
- `config/settings.py`: env-based configuration
- `models/asr_model.py`: ASR wrapper and backend switching
- `models/language_model.py`: beam/LM decoder
- `models/custom_backend_example.py`: custom backend template
- `utils/audio.py`: audio preprocessing utilities

## Development

Run locally:

```bash
python app.py
```

If you change config values, restart the app to reload settings.
