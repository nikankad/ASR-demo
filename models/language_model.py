"""Beam search decoder using pyctcdecode."""

import torch
import logging
from pathlib import Path
from config.settings import BEAM_WIDTH
from config.settings import VOCAB

logger = logging.getLogger(__name__)


class LanguageModel:
    """Beam search decoder with optional KenLM scoring."""

    def __init__(
        self,
        model_name="gpt2",
        model_path=None,
        device="cpu",
        alpha=0.5,
        beta=1.5,
        beam_width=BEAM_WIDTH,
        unk_score_offset=-10.0,
        vocab=None,
    ):
        """Initialize beam search decoder.

        Args:
            model_name: Model name or local LM path
            model_path: Local KenLM ARPA/BIN path
            device: "cpu" or "cuda"
            beam_width: Beam width for search
        """
        self.device = device
        self.model_name = model_name
        self.model_path = model_path
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.model = None
        self.decoder = None
        self.has_external_lm = False
        self.loaded_lm_path = None
        self.beam_width = beam_width
        effective_vocab = vocab if vocab is not None else list(VOCAB)
        if isinstance(effective_vocab, str):
            effective_vocab = list(effective_vocab)
        self.vocab = list(effective_vocab) + [""]
        self._load_model()

    def _resolve_lm_path(self):
        """Resolve LM path robustly for different launch directories."""
        candidate = self.model_path or self.model_name
        if not candidate:
            return None

        p = Path(candidate)
        if p.exists():
            return str(p.resolve())

        repo_root = Path(__file__).resolve().parents[1]
        p2 = repo_root / candidate
        if p2.exists():
            return str(p2.resolve())

        return None

    def _load_unigrams_for_lm(self, lm_path: Path):
        """Load sidecar ARPA unigrams when decoding with KenLM binary files."""
        if lm_path.suffix.lower() == ".arpa":
            return None

        arpa_candidate = lm_path.with_suffix(".arpa")
        if not arpa_candidate.exists():
            logger.warning(
                "Using binary LM without sidecar ARPA unigram source. "
                "Provide a matching .arpa file to improve pyctcdecode accuracy."
            )
            return None

        unigrams = []
        in_one_grams = False
        skip_tokens = {"<s>", "</s>", "<unk>", "<UNK>"}

        with arpa_candidate.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == "\\1-grams:":
                    in_one_grams = True
                    continue
                if in_one_grams and line.startswith("\\2-grams:"):
                    break
                if not in_one_grams or not line or line.startswith("\\"):
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                token = parts[1].strip()
                if token in skip_tokens:
                    continue
                unigrams.append(token)

        if not unigrams:
            logger.warning(
                "No unigrams extracted from %s; decoding may be degraded.",
                arpa_candidate,
            )
            return None

        logger.info("Loaded %d unigrams from %s", len(unigrams), arpa_candidate)
        return unigrams

    def _load_model(self):
        """Initialize pyctcdecode beam search decoder."""
        try:
            from pyctcdecode import build_ctcdecoder

            lm_path = self._resolve_lm_path()
            if lm_path:
                lm_path = Path(lm_path)
                unigrams = self._load_unigrams_for_lm(lm_path)
                self.decoder = build_ctcdecoder(
                    self.vocab,
                    str(lm_path),
                    unigrams=unigrams,
                    alpha=self.alpha,
                    beta=self.beta,
                    unk_score_offset=self.unk_score_offset,
                )
                self.has_external_lm = True
                self.loaded_lm_path = str(lm_path)
                logger.info(f"✓ Loaded beam search decoder with KenLM: {lm_path}")
            else:
                self.decoder = build_ctcdecoder(self.vocab)
                self.has_external_lm = False
                requested = self.model_path or self.model_name
                if requested:
                    logger.warning(f"LM path not found, using decoder without LM: {requested}")
                else:
                    logger.info("LM path not provided, using decoder without LM")
            self.model = True

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
            if logits.dim() != 3:
                logger.error(f"Unexpected logits rank for beam decode: {logits.shape}")
                return None

            vocab_size = len(self.vocab)
            if logits.shape[1] == vocab_size:
                # (batch, n_classes, time) -> (time, n_classes)
                log_probs = logits.squeeze(0).log_softmax(dim=0).T.cpu().numpy()
            elif logits.shape[2] == vocab_size:
                # (batch, time, n_classes) -> (time, n_classes)
                log_probs = logits.squeeze(0).log_softmax(dim=1).cpu().numpy()
            else:
                logger.error(
                    "Cannot infer class dimension for beam decode. "
                    f"Logits shape: {tuple(logits.shape)}, expected vocab size: {vocab_size}"
                )
                return None

            text = self.decoder.decode(log_probs, beam_width=self.beam_width)
            logger.info(f"Beam search decoded: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Beam search decode error: {e}")
            return None

    def correct_text(self, text, max_length=100, temperature=0.7):
        """Dummy method for compatibility."""
        return text, 0.95
