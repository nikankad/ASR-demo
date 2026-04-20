# ASR Live Demo

A Gradio-based web interface for automatic speech recognition with real-time transcription and beam search decoding. Originally built for the IBNet acoustic model, this demo is designed to work with any PyTorch-based ASR model.

## Features

- **Real-time transcription** — Upload audio files or record directly from microphone
- **Beam search decoding** — Improved accuracy over greedy decoding via `pyctcdecode`
- **Minimal dark UI** — Clean, focused interface for model demonstration
- **Confidence scores** — Display model confidence for transcriptions
- **Flexible model support** — Works with any PyTorch-based ASR model

## Setup

### Requirements

- Python 3.11+
- PyTorch, TorchAudio
- Gradio
- pyctcdecode

### Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

The app launches at `http://localhost:7860`

## Usage

1. **Upload or record audio** — Use the audio input component
2. **Toggle beam search** — Enable "Use Language Model" for improved transcription
3. **Transcribe** — Click the Transcribe button
4. **View results** — Transcription text and confidence score displayed

## Architecture

- **[app.py](app.py)** — Gradio interface and inference pipeline
- **[models/asr_model.py](models/asr_model.py)** — Acoustic model loading and inference
- **[models/language_model.py](models/language_model.py)** — Beam search decoder via pyctcdecode
- **[config/settings.py](config/settings.py)** — Configuration (model paths, sample rates, UI settings)
- **[utils/audio.py](utils/audio.py)** — Audio preprocessing (resampling, normalization)

## Model Setup

Place your model weights in `models/model_weights/`:

- `best.pt` — Acoustic model checkpoint (PyTorch)
- `3-gram.pruned.3e-7.arpa` — 3-gram language model (optional, for future use)

Update `config/settings.py` to point to your model paths.

## Customization

### Changing the Model

Edit `config/settings.py`:
```python
MODEL_WEIGHTS_PATH = "path/to/your/model.pt"
```

### Adjusting Audio Settings

```python
SAMPLE_RATE = 16000  # Change to match your model's expected rate
AUDIO_MAX_DURATION = 30  # Max recording length in seconds
```

### UI Customization

Modify the Gradio theme and layout in `app.py`:
```python
with gr.Blocks(theme=gr.themes.Default(...)) as demo:
    # Add/modify UI components here
```

## Development

Edit `app.py` and reload for changes:

```bash
python app.py --debug
```

## Notes

- Audio is automatically resampled to match the configured sample rate
- Beam search decoding works without language model compilation
- The `.arpa` language model file requires `pypi-kenlm` (optional, has build issues on ARM Macs)
