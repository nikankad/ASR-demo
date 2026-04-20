# IBNet ASR Demo

A Gradio-based web interface for automatic speech recognition using the IBNet acoustic model with beam search decoding.

## Features

- **Real-time transcription** — Upload audio files or record directly from microphone
- **Beam search decoding** — Improved accuracy over greedy decoding via `pyctcdecode`
- **Minimal dark UI** — Clean, focused interface for ASR demonstration
- **Confidence scores** — Display model confidence for transcriptions

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
2. **Toggle beam search** — Enable "Use Language Model" for better transcription
3. **Transcribe** — Click the Transcribe button
4. **View results** — Transcription text and confidence score displayed

## Architecture

- **[app.py](app.py)** — Gradio interface and inference pipeline
- **[models/asr_model.py](models/asr_model.py)** — IBNet acoustic model loading and inference
- **[models/language_model.py](models/language_model.py)** — Beam search decoder via pyctcdecode
- **[config/settings.py](config/settings.py)** — Configuration (model paths, sample rates, UI settings)
- **[utils/audio.py](utils/audio.py)** — Audio preprocessing (resampling, normalization)

## Model Weights

Place model weights in `models/model_weights/`:

- `best.pt` — IBNet acoustic model checkpoint
- `3-gram.pruned.3e-7.arpa` — 3-gram language model (optional, for future use)

## Development

To modify the UI or inference logic, edit `app.py` directly. The app supports hot-reloading:

```bash
python app.py --debug
```

## Notes

- Beam search decoding provides better accuracy than greedy decoding without requiring language model compilation
- The `.arpa` language model file requires `pypi-kenlm`, which has build issues on ARM Macs; beam search alone is used as the decoding strategy
- Audio is resampled to 16 kHz and normalized before inference
