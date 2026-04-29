import argparse
from pathlib import Path
import sys
import logging

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

try:
    from models.asr_model import IBNet
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models.asr_model import IBNet


logger = logging.getLogger(__name__)


chars = "abcdefghijklmnopqrstuvwxyz '"
blank = len(chars)
idx2char = {i: c for i, c in enumerate(chars)}
spec_transform = nn.Sequential(
    MelSpectrogram(n_fft=400, sample_rate=16000, hop_length=160, n_mels=64),
    AmplitudeToDB(stype="power", top_db=80),
)


def _ctc_greedy_decode(token_ids):
    decoded = []
    prev = None
    for token in token_ids:
        token = int(token)
        if token != blank and token != prev:
            decoded.append(idx2char.get(token, ""))
        prev = token
    return "".join(decoded)


def _build_lm_decoder(lm_path: Path, alpha: float = 0.5, beta: float = 1.5):
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as exc:
        raise ImportError(
            "pyctcdecode is not installed. Run: pip install pyctcdecode pypi-kenlm"
        ) from exc

    vocab = list(chars)
    vocab.append("")

    unigrams = _load_unigrams_for_lm(lm_path)

    return build_ctcdecoder(
        vocab,
        str(lm_path),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
        unk_score_offset=-10.0,
    )


def _load_unigrams_for_lm(lm_path: Path):
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


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = IBNet(
        n_mels=config.get("n_mels", 64),
        n_classes=config.get("n_classes", 29),
        R=config.get("R", 3),
        expand=2,
        C=192,
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def transcribe_audio(
    audio_path: Path,
    checkpoint_path: Path,
    lm_path: Path = None,
    alpha: float = 0.5,
    beta: float = 1.5,
    beam_width: int = 100,
    device: str = "auto",
) -> dict:
    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    model = _load_model(checkpoint_path, resolved_device)
    feature_transform = spec_transform.to(resolved_device)

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = F.resample(waveform, sample_rate, 16000)

    with torch.no_grad():
        features = feature_transform(waveform.to(resolved_device)).squeeze(0)
        logits = model(features.unsqueeze(0))

    token_ids = logits.argmax(dim=1).squeeze(0).tolist()
    results = {"greedy": _ctc_greedy_decode(token_ids)}

    if lm_path is not None:
        decoder = _build_lm_decoder(lm_path, alpha=alpha, beta=beta)
        log_probs = logits.squeeze(0).log_softmax(dim=0).T.cpu().numpy()
        results["lm"] = decoder.decode(log_probs, beam_width=beam_width)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with IBNet using greedy and/or LM beam search"
    )
    parser.add_argument("--audio", required=True, help="Path to input .wav file")
    parser.add_argument(
        "--checkpoint",
        default="models/model_weights/best.pt",
        help="Path to IBNet checkpoint",
    )
    parser.add_argument(
        "--lm",
        "--arpa",
        dest="lm",
        default=None,
        help="Path to KenLM file (.arpa or .bin). If omitted, only greedy decoding is run.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="LM weight")
    parser.add_argument("--beta", type=float, default=1.5, help="Word insertion bonus")
    parser.add_argument("--beam-width", type=int, default=100, help="Beam width")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    checkpoint_path = Path(args.checkpoint)
    lm_path = Path(args.lm) if args.lm else None

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if lm_path and not lm_path.exists():
        raise FileNotFoundError(f"KenLM file not found: {lm_path}")

    results = transcribe_audio(
        audio_path=audio_path,
        checkpoint_path=checkpoint_path,
        lm_path=lm_path,
        alpha=args.alpha,
        beta=args.beta,
        beam_width=args.beam_width,
        device=args.device,
    )

    print("\n=== Greedy (no LM) ===")
    print(results["greedy"])

    if "lm" in results:
        print("\n=== Beam search + LM ===")
        print(results["lm"])
        print()


if __name__ == "__main__":
    main()
