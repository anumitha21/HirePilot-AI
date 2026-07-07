from pathlib import Path


class SpeechToTextError(RuntimeError):
    """Raised when speech transcription fails."""


class WhisperSpeechToText:
    def __init__(self, model_size: str, device: str, compute_type: str, language: str = "en") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def transcribe(self, audio_path: Path) -> str:
        try:
            segments, _ = self._load_model().transcribe(
                str(audio_path),
                language=self.language,
                vad_filter=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            return transcript
        except Exception as exc:
            raise SpeechToTextError(f"Failed to transcribe audio file: {audio_path}") from exc

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SpeechToTextError(
                "faster-whisper is missing. Install project dependencies with `python -m pip install -e .`."
            ) from exc

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model
