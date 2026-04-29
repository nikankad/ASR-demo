"""ASR model wrapper for inference."""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from importlib import import_module

from config.settings import MODEL_BACKEND, MODEL_CLASS, VOCAB

logger = logging.getLogger(__name__)


DEFAULT_CHARS = VOCAB


class IBConv(nn.Module):
    """Inverted Bottleneck 1D Time-Channel Separable Convolution Module."""

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, expand=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        mid_channel = in_channel * expand

        self.net = nn.Sequential(
            nn.Conv1d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(),
            nn.Conv1d(mid_channel, mid_channel, kernel_size, stride, padding,
                      groups=mid_channel, bias=False),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(),
            nn.Conv1d(mid_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

        self.use_residual = (in_channel == out_channel and stride == 1)

    def forward(self, x):
        out = self.net(x)
        if self.use_residual:
            out = out + x
        return out


class IBBlock(nn.Module):
    """Block of R inverted bottleneck modules with block-level residual."""

    def __init__(self, in_channel, out_channel, kernel_size, R=3, expand=2):
        super().__init__()

        self.layer1 = IBConv(in_channel, out_channel, kernel_size, expand=expand)
        self.layers = nn.ModuleList([
            IBConv(out_channel, out_channel, kernel_size, expand=expand) for _ in range(R - 1)
        ])

        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        out = self.layer1(x)
        for layer in self.layers:
            out = layer(out)
        out = out + self.residual(x)
        return torch.relu(out)


class IBNet(nn.Module):
    """IBNet acoustic model."""

    def __init__(self, n_mels=64, n_classes=29, R=3, expand=2, C=192):
        super().__init__()
        C2 = C * 2
        C3 = C2 * 2
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, C, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(),
            IBBlock(C, C, kernel_size=33, R=R, expand=expand),
            IBBlock(C, C, kernel_size=39, R=R, expand=expand),
            IBBlock(C, C2, kernel_size=51, R=R, expand=expand),
            IBBlock(C2, C2, kernel_size=63, R=R, expand=expand),
            IBBlock(C2, C2, kernel_size=75, R=R, expand=expand),
            IBConv(C2, C2, kernel_size=87, expand=expand),
            nn.Conv1d(C2, C3, kernel_size=1, bias=False),
            nn.BatchNorm1d(C3),
            nn.ReLU(),
            nn.Conv1d(C3, n_classes, dilation=2, kernel_size=1),
        )

    def forward(self, x):
        x = x.squeeze(1)
        return self.net(x)


class ASRModel:
    """Speech recognition model wrapper for IBNet."""

    def __init__(self, model_path, device="cpu"):
        """Initialize model.

        Args:
            model_path: Path to model weights
            device: "cpu" or "cuda"
        """
        self.device = device
        self.model_path = Path(model_path)
        self.model = None
        self.chars = DEFAULT_CHARS
        self.blank_idx = len(self.chars)
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.idx2char[self.blank_idx] = ""
        self.backend = MODEL_BACKEND.lower()
        self.custom_model = None
        self.spec_transform = None
        self._load_model()

    @property
    def vocab(self):
        return list(self.chars) + [""]

    def _set_vocab(self, vocab):
        if isinstance(vocab, list):
            vocab = "".join(vocab)
        if isinstance(vocab, str) and vocab.strip():
            self.chars = vocab
            self.blank_idx = len(self.chars)
            self.idx2char = {i: c for i, c in enumerate(self.chars)}
            self.idx2char[self.blank_idx] = ""

    def _load_model(self):
        """Load backend model weights from disk."""
        logger.debug(f"Loading ASR model from: {self.model_path}")

        if self.backend == "python_class":
            self._load_custom_python_model()
            return

        if not self.model_path.exists():
            logger.error(f"Model weights not found at {self.model_path}")
            return

        try:
            logger.debug(f"Loading checkpoint...")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {})
            logger.debug(f"Checkpoint config: {config}")

            if self.backend != "ibnet":
                logger.warning(
                    "Unsupported MODEL_BACKEND '%s'. Supported: ibnet, python_class. Falling back to 'ibnet'.",
                    self.backend,
                )
                self.backend = "ibnet"

            self._set_vocab(config.get("vocab", self.chars))

            state_dict = checkpoint.get("model_state_dict", checkpoint)
            logger.debug(f"State dict keys: {len(state_dict)} parameters")

            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                logger.debug("Removing '_orig_mod.' prefix from state dict keys")
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            # Initialize IBNet with config
            logger.debug(f"Initializing IBNet with config: n_mels={config.get('n_mels', 64)}, n_classes={config.get('n_classes', 29)}")
            self.model = IBNet(
                n_mels=config.get("n_mels", 64),
                n_classes=config.get("n_classes", len(self.chars) + 1),
                R=config.get("R", 3),
                expand=2,
                C=192,
            ).to(self.device)

            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            from utils.audio import spec_transform
            self.spec_transform = spec_transform.to(self.device)
            logger.info(f"✓ IBNet model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.model = None

    def _load_custom_python_model(self):
        """Load user-provided model class from MODEL_CLASS path.

        Expected format: "package.module:ClassName"
        The class must provide: transcribe(audio_waveform) -> dict
        """
        if not MODEL_CLASS:
            logger.error("MODEL_BACKEND=python_class requires MODEL_CLASS to be set")
            self.model = None
            return

        try:
            module_name, class_name = MODEL_CLASS.split(":", maxsplit=1)
            module = import_module(module_name)
            model_cls = getattr(module, class_name)
            self.custom_model = model_cls(model_path=str(self.model_path), device=self.device, vocab=self.chars)

            if hasattr(self.custom_model, "vocab"):
                custom_vocab = getattr(self.custom_model, "vocab")
                if isinstance(custom_vocab, list):
                    custom_vocab = "".join(custom_vocab)
                self._set_vocab(custom_vocab)

            self.model = self.custom_model
            logger.info("✓ Custom model backend loaded from %s", MODEL_CLASS)
        except Exception as e:
            logger.error("Failed to load custom model backend '%s': %s", MODEL_CLASS, e, exc_info=True)
            self.model = None

    def _ctc_greedy_decode(self, token_ids):
        """Greedy CTC decoding."""
        decoded = []
        prev = None
        for token_id in token_ids:
            token = int(token_id)
            if token != self.blank_idx and token != prev:
                decoded.append(self.idx2char.get(token, ""))
            prev = token
        return "".join(decoded)

    def transcribe(self, audio_waveform):
        """Run inference on audio.

        Args:
            audio_waveform: Numpy array of audio samples (16000 Hz)

        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {"text": "Model not loaded", "confidence": 0.0}

        if self.backend == "python_class":
            try:
                result = self.custom_model.transcribe(audio_waveform)
                if isinstance(result, dict):
                    return result
                return {"text": str(result), "confidence": 0.0}
            except Exception as e:
                logger.error("Custom backend inference error: %s", e, exc_info=True)
                return {"text": f"Inference error: {str(e)}", "confidence": 0.0}

        try:
            logger.debug(f"Transcribe input - Shape: {audio_waveform.shape}, Dtype: {audio_waveform.dtype}")

            # Convert to tensor and add batch/channel dims
            audio_tensor = torch.from_numpy(audio_waveform).float().to(self.device)
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
            logger.debug(f"Audio tensor shape after unsqueeze: {audio_tensor.shape}")

            # Compute spectrogram features (spec_transform is on device)
            logger.debug("Computing mel-spectrogram...")
            features = self.spec_transform(audio_tensor).squeeze(0)  # (n_mels, time)
            logger.debug(f"Spectrogram shape: {features.shape}, Range: [{features.min():.2f}, {features.max():.2f}]")

            inputs = features.unsqueeze(0)  # already on device
            logger.debug(f"Model input shape: {inputs.shape}")

            # Run inference
            logger.debug("Running IBNet inference...")
            with torch.no_grad():
                logits = self.model(inputs)
                log_probs = logits.squeeze(0).log_softmax(dim=0)

            logger.debug(f"Logits shape: {logits.shape}, Range: [{logits.min():.4f}, {logits.max():.4f}]")

            # Expected: (batch, n_classes, time) or (batch, time, n_classes)
            if logits.shape[1] < logits.shape[2]:
                # Shape is (batch, n_classes, time) — argmax over dim 1
                logger.debug("Using dim=1 for argmax")
                token_ids = logits.argmax(dim=1).squeeze(0)
            else:
                # Shape is (batch, time, n_classes) — argmax over dim 2
                logger.debug("Using dim=2 for argmax")
                token_ids = logits.argmax(dim=2).squeeze(0)

            logger.debug(f"Token IDs shape: {token_ids.shape if isinstance(token_ids, torch.Tensor) else type(token_ids)}")

            # Ensure it's 1D and convert to list
            if isinstance(token_ids, torch.Tensor):
                if token_ids.dim() > 1:
                    token_ids = token_ids.flatten()
                token_ids = token_ids.cpu().numpy().tolist()
            else:
                token_ids = [int(token_ids)]

            logger.debug(f"Token IDs (first 10): {token_ids[:10] if len(token_ids) >= 10 else token_ids}")

            text = self._ctc_greedy_decode(token_ids)
            logger.debug(f"Decoded text: '{text}'")

            # Compute confidence from log-probs (computed above inside no_grad)
            confidence = float(log_probs.max().item())
            confidence = max(0.0, min(1.0, (confidence + 10) / 10))  # Normalize to [0, 1]

            logger.info(f"Inference result - Text: '{text}', Confidence: {confidence:.4f}")

            return {
                "text": text if text.strip() else "(silence)",
                "confidence": confidence,
                "logits": logits
            }
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return {
                "text": f"Inference error: {str(e)}",
                "confidence": 0.0
            }
