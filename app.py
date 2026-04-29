"""Main Gradio application for ASR demo."""

import torch
import numpy as np
import torchaudio.functional as F
import gradio as gr
import logging
import psutil
from pathlib import Path
from collections import deque
from datetime import datetime
import pandas as pd

from config.settings import (
    MODEL_WEIGHTS_PATH,
    LM_MODEL_NAME,
    LM_MODEL_PATH,
    LM_ALPHA,
    LM_BETA,
    LM_UNK_SCORE_OFFSET,
    BEAM_WIDTH,
    SAMPLE_RATE,
    UI_TITLE,
    UI_DESCRIPTION,
)
from models import ASRModel, LanguageModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("Starting ASR Demo Application")
logger.info("=" * 60)
logger.info(f"Device: {torch.cuda.is_available() and 'CUDA' or 'CPU'}")
logger.info(f"ASR Model Path: {MODEL_WEIGHTS_PATH}")
logger.info(f"LM Model Name: {LM_MODEL_NAME}")
logger.info(f"LM Model Path: {LM_MODEL_PATH}")
logger.info(f"LM Alpha/Beta: {LM_ALPHA}/{LM_BETA}")
logger.info(f"LM Beam Width: {BEAM_WIDTH}")
logger.info(f"Sample Rate: {SAMPLE_RATE} Hz")

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading ASR model on device: {device}")
asr_model = ASRModel(MODEL_WEIGHTS_PATH, device=device)
logger.info(f"ASR model loaded: {asr_model.model is not None}")

logger.info(f"Loading LM model: {LM_MODEL_NAME}")
lm_model = LanguageModel(
    LM_MODEL_NAME,
    model_path=LM_MODEL_PATH,
    device=device,
    alpha=LM_ALPHA,
    beta=LM_BETA,
    beam_width=BEAM_WIDTH,
    unk_score_offset=LM_UNK_SCORE_OFFSET,
    vocab=asr_model.vocab[:-1],
)
logger.info(f"LM model loaded: {lm_model.model is not None}")
logger.info(f"External KenLM loaded: {lm_model.has_external_lm}")
logger.info("=" * 60)

RESOURCE_HISTORY = deque(maxlen=120)


def get_model_info():
    """Return markdown with loaded model specifications."""
    backend = getattr(asr_model, "backend", "unknown")
    vocab_size = len(getattr(asr_model, "vocab", []))
    weights_path = Path(MODEL_WEIGHTS_PATH).resolve()
    lm_path = Path(LM_MODEL_PATH).resolve()

    model_obj = getattr(asr_model, "model", None)
    parameter_count = "N/A"
    model_class = "Not loaded"
    if model_obj is not None and hasattr(model_obj, "parameters"):
        try:
            parameter_count = f"{sum(p.numel() for p in model_obj.parameters()):,}"
            model_class = model_obj.__class__.__name__
        except Exception:
            model_class = type(model_obj).__name__
    elif model_obj is not None:
        model_class = type(model_obj).__name__

    lm_status = "Loaded" if lm_model.model is not None else "Unavailable"
    lm_mode = "KenLM" if lm_model.has_external_lm else "CTC beam only"

    return (
        "### Model Information\n"
        f"- **Backend:** `{backend}`\n"
        f"- **Model class:** `{model_class}`\n"
        f"- **Weights path:** `{weights_path}`\n"
        f"- **Device:** `{device}`\n"
        f"- **Sample rate:** `{SAMPLE_RATE} Hz`\n"
        f"- **Vocabulary size (incl. blank):** `{vocab_size}`\n"
        f"- **Parameters:** `{parameter_count}`\n"
        f"- **Language model:** `{lm_status}` ({lm_mode})\n"
        f"- **LM path:** `{lm_path}`\n"
        f"- **Beam width:** `{BEAM_WIDTH}`\n"
        f"- **LM alpha/beta:** `{LM_ALPHA}` / `{LM_BETA}`"
    )


def _collect_resource_sample():
    """Collect one resource sample for charting."""
    process = psutil.Process()
    mem = process.memory_info()
    sys_mem = psutil.virtual_memory()
    cpu_process = process.cpu_percent(interval=None)
    cpu_system = psutil.cpu_percent(interval=None)
    timestamp = datetime.now().strftime("%H:%M:%S")

    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0
    gpu_label = "Not available"
    if torch.cuda.is_available():
        dev_idx = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        gpu_allocated_mb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
        gpu_reserved_mb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
        gpu_label = dev_name

    sample = {
        "time": timestamp,
        "process_cpu": float(cpu_process),
        "system_cpu": float(cpu_system),
        "process_rss_mb": float(mem.rss / (1024 ** 2)),
        "system_ram_pct": float(sys_mem.percent),
        "gpu_allocated_mb": float(gpu_allocated_mb),
        "gpu_reserved_mb": float(gpu_reserved_mb),
        "gpu_label": gpu_label,
    }
    RESOURCE_HISTORY.append(sample)
    return sample


def _build_resource_outputs():
    """Build chart datasets and status markdown from history."""
    if not RESOURCE_HISTORY:
        _collect_resource_sample()

    df = pd.DataFrame(list(RESOURCE_HISTORY))
    cpu_df = df[["time", "process_cpu", "system_cpu"]].melt(
        id_vars="time",
        var_name="metric",
        value_name="value",
    )
    ram_df = df[["time", "process_rss_mb", "system_ram_pct"]].melt(
        id_vars="time",
        var_name="metric",
        value_name="value",
    )
    gpu_df = df[["time", "gpu_allocated_mb", "gpu_reserved_mb"]].melt(
        id_vars="time",
        var_name="metric",
        value_name="value",
    )

    latest = RESOURCE_HISTORY[-1]
    gpu_note = f"`{latest['gpu_label']}`" if torch.cuda.is_available() else "`Not available`"

    status_md = (
        "### Live Resource Usage\n"
        f"- **Last sample:** `{latest['time']}`\n"
        f"- **Process CPU:** `{latest['process_cpu']:.1f}%`\n"
        f"- **System CPU:** `{latest['system_cpu']:.1f}%`\n"
        f"- **Process RSS memory:** `{latest['process_rss_mb']:.1f} MB`\n"
        f"- **System RAM used:** `{latest['system_ram_pct']:.1f}%`\n"
        f"- **GPU:** {gpu_note}"
    )

    return status_md, cpu_df, ram_df, gpu_df


def refresh_resource_charts():
    """Collect sample and refresh all resource chart outputs."""
    _collect_resource_sample()
    return _build_resource_outputs()


def transcribe_audio(audio_input):
    """Gradio interface for audio transcription.

    Args:
        audio_input: Tuple of (sample_rate, waveform) from Gradio audio component
    Returns:
        Tuple of (raw_transcription_text, lm_transcription_text)
    """
    logger.info("-" * 60)
    logger.info("Transcription Request Started")
    try:
        if audio_input is None:
            logger.warning("No audio provided")
            return "No audio provided", "No audio provided"

        sr, waveform = audio_input
        logger.debug(f"Input audio - Sample rate: {sr} Hz, Shape: {waveform.shape}")

        # Match script behavior: convert PCM integers to float32 in [-1, 1]
        if np.issubdtype(waveform.dtype, np.integer):
            max_abs = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)
            waveform = waveform.astype("float32") / float(max_abs)
        else:
            waveform = waveform.astype("float32")

        logger.debug(f"Audio dtype: {waveform.dtype}, Range: [{waveform.min():.4f}, {waveform.max():.4f}]")

        # Handle 2D arrays (stereo)
        if len(waveform.shape) > 1:
            logger.debug("Converting stereo to mono")
            waveform = waveform.mean(axis=1)

        # Resample if necessary (match transcribe_lm.py with torchaudio)
        if sr != SAMPLE_RATE:
            logger.debug(f"Resampling from {sr} Hz to {SAMPLE_RATE} Hz")
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            waveform = F.resample(waveform_tensor, sr, SAMPLE_RATE).squeeze(0).numpy()

        # Run ASR inference
        logger.info("Running ASR inference")
        result = asr_model.transcribe(waveform)
        raw_text = result.get("text", "")
        asr_confidence = result.get("confidence", 0.0)
        logits = result.get("logits")
        logger.info(f"ASR Result - Text: '{raw_text}', Confidence: {asr_confidence:.4f}")

        if lm_model.model is not None and logits is not None:
            logger.info("Applying Beam Search Decoder with LM")
            lm_text = lm_model.decode_beam_search(logits)
            if lm_text:
                if lm_text == raw_text:
                    logger.info("Beam/LM decode matched greedy output")
                else:
                    logger.info("Beam/LM decode changed output")
            else:
                logger.warning("LM decode failed, using raw output")
                lm_text = raw_text
        else:
            logger.warning("LM unavailable or logits unavailable")
            lm_text = "Model not loaded"

        logger.info(f"Final Output - Raw: '{raw_text}' | LM: '{lm_text}'")
        logger.info("-" * 60)
        return raw_text, lm_text

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        logger.info("-" * 60)
        return error_msg, error_msg


# Create Gradio interface with minimal dark theme
with gr.Blocks(theme=gr.themes.Default(primary_hue="slate", text_size="sm"), title="ASR demo", css="""
.share-button { display: none !important; }
.progress-bar-wrap { display: none !important; }
.transcribe-btn {
    transition: transform 0.12s ease, box-shadow 0.2s ease;
}
.transcribe-btn:hover {
    transform: translateY(-1px);
}
.transcribe-btn:active {
    transform: translateY(0);
}
.transcribe-btn.generating,
.transcribe-btn.pending,
.transcribe-btn.loading {
    box-shadow: 0 0 0 0 rgba(148, 163, 184, 0.45);
    animation: pulse 1.2s ease-out infinite;
}
.transcribe-btn.generating::after,
.transcribe-btn.pending::after,
.transcribe-btn.loading::after {
    content: '';
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-left: 8px;
    border: 2px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(148, 163, 184, 0.45); }
    70% { box-shadow: 0 0 0 8px rgba(148, 163, 184, 0); }
    100% { box-shadow: 0 0 0 0 rgba(148, 163, 184, 0); }
}
@keyframes spin { to { transform: rotate(360deg); } }
""") as demo:
    gr.Markdown(f"# {UI_TITLE}")
    gr.Markdown(f"*{UI_DESCRIPTION}*")

    lm_loaded = lm_model.model is not None
    lm_status = "Loaded" if lm_loaded else "Model not loaded"

    with gr.Row():
        audio_input = gr.Audio(
            label="Upload or Record Audio",
            type="numpy",
            sources=["upload", "microphone"]
        )

    with gr.Row():
        transcribe_btn = gr.Button("Transcribe", variant="primary", elem_classes=["transcribe-btn"])

    with gr.Row():
        raw_text_output = gr.Textbox(
            label="Raw (Greedy CTC)",
            interactive=False,
            lines=3
        )
        lm_text_output = gr.Textbox(
            label=f"With Language Model ({lm_status})",
            interactive=False,
            lines=3,
            value="" if lm_loaded else "Model not loaded"
        )

    with gr.Row():
        with gr.Accordion("Model Panel", open=True):
            with gr.Tabs(selected="resource-usage"):
                with gr.Tab("Model Information", id="model-info"):
                    model_info_output = gr.Markdown(get_model_info())
                with gr.Tab("Resource Usage", id="resource-usage"):
                    resource_usage_output = gr.Markdown()
                    cpu_chart = gr.LinePlot(
                        x="time",
                        y="value",
                        color="metric",
                        title="CPU Usage (%)",
                        y_title="Percent",
                        x_title="Time",
                        height=240,
                    )
                    memory_chart = gr.LinePlot(
                        x="time",
                        y="value",
                        color="metric",
                        title="RAM Usage",
                        y_title="Value",
                        x_title="Time",
                        height=240,
                    )
                    gpu_chart = gr.LinePlot(
                        x="time",
                        y="value",
                        color="metric",
                        title="GPU Memory (MB)",
                        y_title="MB",
                        x_title="Time",
                        height=240,
                    )
                    refresh_usage_btn = gr.Button("Refresh Usage")

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[raw_text_output, lm_text_output],
        show_progress="hidden"
    )

    refresh_usage_btn.click(
        fn=refresh_resource_charts,
        inputs=None,
        outputs=[resource_usage_output, cpu_chart, memory_chart, gpu_chart],
        show_progress="hidden",
    )

    model_refresh_timer = gr.Timer(2.0)
    model_refresh_timer.tick(
        fn=refresh_resource_charts,
        inputs=None,
        outputs=[resource_usage_output, cpu_chart, memory_chart, gpu_chart],
        show_progress="hidden",
    )

    demo.load(
        fn=refresh_resource_charts,
        inputs=None,
        outputs=[resource_usage_output, cpu_chart, memory_chart, gpu_chart],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="config/favicon.ico"
    )
