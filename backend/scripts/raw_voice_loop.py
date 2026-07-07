import logging
from pathlib import Path

from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.voice.audio_io import LocalAudioIO
from backend.voice.stt import WhisperSpeechToText
from backend.voice.tts import EdgeTextToSpeech

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    audio_io = LocalAudioIO(sample_rate=settings.voice_sample_rate)
    speech_to_text = WhisperSpeechToText(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
    )
    text_to_speech = EdgeTextToSpeech(voice=settings.voice_tts_voice)

    logger.info("M0 raw voice loop starting.")
    logger.info(
        "Recording until you stop speaking, up to %s seconds. Speak now.",
        settings.voice_max_record_seconds,
    )

    recorded_path: Path | None = None
    spoken_path: Path | None = None

    try:
        recorded_path = audio_io.record_until_silence(
            max_duration_seconds=settings.voice_max_record_seconds,
            silence_seconds=settings.voice_silence_seconds,
            vad_threshold=settings.voice_vad_threshold,
        )
        logger.info("Transcribing recorded audio.")
        transcript = speech_to_text.transcribe(recorded_path)

        if not transcript:
            transcript = "I could not hear anything clearly. Please try again."
            logger.warning("No clear transcript detected.")
        else:
            logger.info("Transcript: %s", transcript)

        spoken_path = text_to_speech.synthesize_to_file(transcript)
        logger.info("Playing synthesized echo.")
        audio_io.play_audio_file(spoken_path)
    finally:
        for path in (recorded_path, spoken_path):
            if path and path.exists():
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
