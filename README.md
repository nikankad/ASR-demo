# ASR Live Demo

A Gradio-based web interface for automatic speech recognition with real-time transcription and beam search decoding. Originally built for the IBNet acoustic model, this demo is designed to work with any PyTorch-based ASR model.

## Features

- Real-time transcription: Upload audio files or record directly from microphone
- Beam search decoding: Improved accuracy over greedy decoding via `pyctcdecode`
- Minimal dark UI: Clean, focused interface for model demonstration
- Confidence scores: Display model confidence for transcriptions
- Flexible model support: Works with any PyTorch-based ASR model
- Audio preprocessing: Automatic resampling and normalization
- Interactive controls: Toggle beam search and adjust decoding parameters

## Setup

### Requirements

- Python 3.11 or higher
- PyTorch and TorchAudio
- Gradio for web interface
- pyctcdecode for beam search decoding
- librosa for audio processing
- numpy for numerical operations

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

1. Upload or record audio using the audio input component
2. Toggle "Use Language Model" to enable beam search decoding for improved transcription
3. Click the Transcribe button to process the audio
4. View results including transcription text and confidence score

## Architecture

- `app.py`: Main Gradio interface and inference pipeline
- `models/asr_model.py`: Acoustic model loading and CTC decoding
- `models/language_model.py`: Beam search decoder via pyctcdecode
- `config/settings.py`: Configuration for model paths, sample rates, and UI settings
- `utils/audio.py`: Audio preprocessing utilities (resampling, normalization, mel-spectrograms)

## Model Setup

Place your model weights in `models/model_weights/`:

- `best.pt`: Acoustic model checkpoint (PyTorch format)
- `3-gram.pruned.3e-7.arpa`: 3-gram language model file (optional, for future use)

Update `config/settings.py` (or env vars) to point to your model paths and adjust sample rates to match your model.

### Open-source BYO model mode

You can plug in your own model without editing core app code.

1. Implement a backend class with:
   - `__init__(self, model_path, device="cpu", vocab=None)`
   - `transcribe(self, audio_waveform)` returning a dict with at least `{"text": ...}`
2. Use `models/custom_backend_example.py` as a template.
3. Set env vars before running:

```bash
export MODEL_BACKEND=python_class
export MODEL_CLASS=models.custom_backend_example:CustomASRBackend
export MODEL_WEIGHTS_PATH=/absolute/or/relative/path/to/your/model.pt
export VOCAB="abcdefghijklmnopqrstuvwxyz '"
python app.py
```

If your backend also returns `logits` in CTC shape, the app can still run beam search decoding.

## Customization

### Changing the Model

Edit `config/settings.py` to point to your model:

```python
MODEL_WEIGHTS_PATH = "path/to/your/model.pt"
LM_MODEL_PATH = "path/to/your/lm.arpa"  # optional
```

The demo automatically detects model architecture and loads the appropriate weights.

### Adjusting Audio Settings

```python
SAMPLE_RATE = 16000  # Change to match your model's expected rate
AUDIO_MAX_DURATION = 30  # Max recording length in seconds
```

### UI Customization

Modify the Gradio theme and layout in `app.py`:

```python
with gr.Blocks(theme=gr.themes.Default(...)) as demo:
    gr.Markdown(f"# {UI_TITLE}")
    # Add/modify UI components here
```

### Beam Search Parameters

Adjust beam search settings in `config/settings.py`:

```python
BEAM_WIDTH = 100  # Number of beams to maintain
ALPHA = 0.5       # Language model weight (if using LM)
BETA = 1.5        # Word insertion bonus
```

## Development

Edit `app.py` and the app will hot-reload on changes:

```bash
python app.py --debug
```

Logs show inference details, model loading status, and timing information for debugging.

## Technical Details

### Model Support

Currently tested with CTC-based acoustic models (like IBNet). The pipeline expects:
- Model output shape: (batch, n_classes, time)
- CTC blank index at position len(vocab)
- PyTorch `.pt` checkpoint format

### Audio Processing

1. Loads audio at any sample rate
2. Converts to mono if stereo
3. Resamples to configured sample rate (default 16kHz)
4. Normalizes amplitude
5. Computes mel-spectrogram features

### Decoding

- Greedy decoding: Takes highest-probability character at each timestep
- Beam search: Explores multiple hypotheses to find better transcriptions
- No language model needed for beam search to work

## Notes

- Audio is automatically resampled to match the configured sample rate
- Beam search decoding works standalone without language model compilation
- The `.arpa` language model file requires `pypi-kenlm` to integrate (optional, has build issues on ARM Macs)
- Confidence scores are normalized to [0, 1] range
- The app supports both file upload and live microphone recording
