"""Example plug-in backend for BYO ASR models.

Set in environment:
MODEL_BACKEND=python_class
MODEL_CLASS=models.custom_backend_example:CustomASRBackend
"""


class CustomASRBackend:
    """Minimal interface expected by ASRModel python_class backend."""

    def __init__(self, model_path, device="cpu", vocab=None):
        self.model_path = model_path
        self.device = device
        self.vocab = vocab or "abcdefghijklmnopqrstuvwxyz '"
        # Load your own model here

    def transcribe(self, audio_waveform):
        """Return dict with at least 'text' key.

        Optional keys:
        - confidence: float in [0, 1]
        - logits: tensor/array shaped for CTC decoding (if you want LM decoding)
        """
        return {
            "text": "replace with your model inference",
            "confidence": 0.0,
        }
