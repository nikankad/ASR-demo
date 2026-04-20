"""Beam search decoder using pyctcdecode."""

import torch
import logging

logger = logging.getLogger(__name__)


class LanguageModel:
    """Beam search decoder (no LM)."""

    def __init__(self, model_name="gpt2", model_path=None, device="cpu", alpha=0.5, beta=1.5, beam_width=100):
        """Initialize beam search decoder.

        Args:
            model_name: Unused
            model_path: Unused (LM files don't compile on ARM)
            device: "cpu" or "cuda"
            beam_width: Beam width for search
        """
        self.device = device
        self.model = None
        self.decoder = None
        self.beam_width = beam_width
        self._load_model()

    def _load_model(self):
        """Initialize pyctcdecode beam search decoder."""
        try:
            from pyctcdecode import build_ctcdecoder

            chars = "abcdefghijklmnopqrstuvwxyz '"
            vocab = list(chars)
            vocab.append("")  # blank at index 28

            self.decoder = build_ctcdecoder(vocab)
            self.model = True
            logger.info("✓ Loaded beam search decoder (pyctcdecode)")

        except ImportError:
            logger.error("pyctcdecode not installed. Run: pip install pyctcdecode")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load decoder: {e}", exc_info=True)
            self.model = None

    def decode_beam_search(self, logits):
        """Decode logits using beam search.

        Args:
            logits: Model output (1, n_classes, time)

        Returns:
            Decoded text
        """
        if not self.decoder:
            return None

        try:
            log_probs = logits.squeeze(0).log_softmax(dim=0).T.cpu().numpy()
            text = self.decoder.decode(log_probs, beam_width=self.beam_width)
            logger.info(f"Beam search decoded: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Beam search decode error: {e}")
            return None

    def correct_text(self, text, max_length=100, temperature=0.7):
        """Dummy method for compatibility."""
        return text, 0.95
