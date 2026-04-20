"""Main Gradio application for ASR demo."""

import torch
import gradio as gr
import logging

from config.settings import (
    MODEL_WEIGHTS_PATH, LM_MODEL_NAME, LM_MODEL_PATH, SAMPLE_RATE, UI_TITLE, UI_DESCRIPTION
)
from models import ASRModel, LanguageModel
from utils.audio import normalize_audio

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
logger.info(f"Sample Rate: {SAMPLE_RATE} Hz")

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading ASR model on device: {device}")
asr_model = ASRModel(MODEL_WEIGHTS_PATH, device=device)
logger.info(f"ASR model loaded: {asr_model.model is not None}")

logger.info(f"Loading LM model: {LM_MODEL_NAME}")
lm_model = LanguageModel(LM_MODEL_NAME, model_path=LM_MODEL_PATH, device=device)
logger.info(f"LM model loaded: {lm_model.model is not None}")
logger.info("=" * 60)


def transcribe_audio(audio_input, use_lm):
    """Gradio interface for audio transcription.

    Args:
        audio_input: Tuple of (sample_rate, waveform) from Gradio audio component
        use_lm: Boolean to enable language model post-processing

    Returns:
        Tuple of (transcription_text, confidence)
    """
    logger.info("-" * 60)
    logger.info("Transcription Request Started")
    try:
        if audio_input is None:
            logger.warning("No audio provided")
            return "No audio provided", 0.0

        sr, waveform = audio_input
        logger.debug(f"Input audio - Sample rate: {sr} Hz, Shape: {waveform.shape}")

        # Convert to float
        waveform = waveform.astype('float32')
        logger.debug(f"Audio dtype: {waveform.dtype}, Range: [{waveform.min():.4f}, {waveform.max():.4f}]")

        # Handle 2D arrays (stereo)
        if len(waveform.shape) > 1:
            logger.debug("Converting stereo to mono")
            waveform = waveform.mean(axis=1)

        # Resample if necessary
        if sr != SAMPLE_RATE:
            logger.debug(f"Resampling from {sr} Hz to {SAMPLE_RATE} Hz")
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Normalize
        logger.debug("Normalizing audio")
        waveform = normalize_audio(waveform)
        logger.debug(f"After normalization - Range: [{waveform.min():.4f}, {waveform.max():.4f}]")

        # Run ASR inference
        logger.info("Running ASR inference")
        result = asr_model.transcribe(waveform)
        text = result.get("text", "")
        asr_confidence = result.get("confidence", 0.0)
        logits = result.get("logits")
        logger.info(f"ASR Result - Text: '{text}', Confidence: {asr_confidence:.4f}")

        # Apply language model if enabled
        if use_lm:
            if lm_model.model is not None and logits is not None:
                logger.info("Applying Beam Search Decoder with LM")
                lm_text = lm_model.decode_beam_search(logits)
                if lm_text:
                    text = lm_text
                    final_confidence = 0.95
                    logger.info(f"LM Result: '{text}'")
                else:
                    logger.warning("LM decode failed, using greedy")
                    final_confidence = asr_confidence
            else:
                logger.warning("LM requested but not loaded or logits unavailable")
                final_confidence = asr_confidence
        else:
            logger.debug("LM not enabled")
            final_confidence = asr_confidence

        logger.info(f"Final Output - Text: '{text}', Confidence: {final_confidence:.4f}")
        logger.info("-" * 60)
        return text, final_confidence

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        logger.info("-" * 60)
        return error_msg, 0.0


# Create Gradio interface with minimal dark theme
with gr.Blocks(theme=gr.themes.Default(primary_hue="slate", text_size="sm")) as demo:
    gr.Markdown(f"# {UI_TITLE}")
    gr.Markdown(f"*{UI_DESCRIPTION}*")

    # Language model status
    lm_status = "✅ Language Model Loaded" if lm_model.model is not None else "⚠️ Language Model Not Loaded"
    gr.Markdown(f"**Status:** {lm_status}")

    with gr.Row():
        audio_input = gr.Audio(
            label="Upload or Record Audio",
            type="numpy",
            sources=["upload", "microphone"]
        )

    with gr.Row():
        use_lm_toggle = gr.Checkbox(
            label="Use Language Model",
            value=False,
            interactive=(lm_model.model is not None)
        )

    with gr.Row():
        transcribe_btn = gr.Button("Transcribe", variant="primary")

    with gr.Row():
        text_output = gr.Textbox(
            label="Transcription",
            interactive=False,
            lines=3
        )
        confidence_output = gr.Number(
            label="Confidence",
            interactive=False
        )

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, use_lm_toggle],
        outputs=[text_output, confidence_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="config/favicon.ico"
    )
