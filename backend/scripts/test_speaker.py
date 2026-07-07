import argparse
import logging
from pathlib import Path

from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.voice.audio_io import LocalAudioIO
from backend.voice.tts import EdgeTextToSpeech

logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    audio_io = LocalAudioIO(sample_rate=settings.voice_sample_rate)
    text_to_speech = EdgeTextToSpeech(voice=settings.voice_tts_voice)
    spoken_path: Path | None = None

    try:
        logger.info("Synthesizing speaker test phrase.")
        spoken_path = text_to_speech.synthesize_to_file(args.text)
        logger.info("Playing speaker test phrase.")
        audio_io.play_audio_file(spoken_path)
        logger.info("Speaker test completed.")
    finally:
        if spoken_path and spoken_path.exists():
            spoken_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Edge TTS and local speaker playback.")
    parser.add_argument(
        "--text",
        default="Hello. This is HirePilot testing your speaker output.",
        help="Text to synthesize and play.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

