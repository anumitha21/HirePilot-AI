import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile


class TextToSpeechError(RuntimeError):
    """Raised when speech synthesis fails."""


class EdgeTextToSpeech:
    def __init__(self, voice: str) -> None:
        self.voice = voice

    def synthesize_to_file(self, text: str) -> Path:
        try:
            return asyncio.run(self._synthesize(text))
        except Exception as exc:
            raise TextToSpeechError("Failed to synthesize speech with Edge TTS.") from exc

    async def _synthesize(self, text: str) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise TextToSpeechError(
                "edge-tts is missing. Install project dependencies with `python -m pip install -e .`."
            ) from exc

        with NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            output_path = Path(audio_file.name)

        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(str(output_path))
        return output_path

