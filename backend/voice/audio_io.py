from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
import time
import logging

logger = logging.getLogger(__name__)


class AudioIOError(RuntimeError):
    """Raised when local audio input or output fails."""


class LocalAudioIO:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def record_to_wav(self, duration_seconds: int) -> Path:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as exc:
            raise AudioIOError(
                "Audio dependencies are missing. Install project dependencies with `python -m pip install -e .`."
            ) from exc

        try:
            frames = duration_seconds * self.sample_rate
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()

            with NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                output_path = Path(audio_file.name)

            sf.write(output_path, recording, self.sample_rate)
            return output_path
        except Exception as exc:
            raise AudioIOError("Failed to record microphone audio.") from exc

    def record_until_silence(
        self,
        max_duration_seconds: int,
        silence_seconds: float,
        vad_threshold: float,
        block_duration_seconds: float = 0.2,
    ) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except ImportError as exc:
            raise AudioIOError(
                "Audio dependencies are missing. Install project dependencies with `python -m pip install -e .`."
            ) from exc

        try:
            block_size = int(self.sample_rate * block_duration_seconds)
            max_blocks = int(max_duration_seconds / block_duration_seconds)
            silence_limit = max(1, int(silence_seconds / block_duration_seconds))
            chunks = []
            has_speech = False
            silent_blocks = 0

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
            ) as stream:
                for _ in range(max_blocks):
                    block, _ = stream.read(block_size)
                    chunks.append(block.copy())
                    rms = float(np.sqrt(np.mean(np.square(block))))

                    if rms >= vad_threshold:
                        has_speech = True
                        silent_blocks = 0
                    elif has_speech:
                        silent_blocks += 1

                    if has_speech and silent_blocks >= silence_limit:
                        break

                    time.sleep(0.01)

            if not chunks:
                raise AudioIOError("No audio was recorded. The microphone may be unavailable or silent.")

            recording = np.concatenate(chunks, axis=0)
            with NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                output_path = Path(audio_file.name)

            sf.write(output_path, recording, self.sample_rate)
            return output_path
        except AudioIOError:
            raise
        except Exception as exc:
            raise AudioIOError("Failed to record microphone audio with VAD.") from exc

    def play_audio_file(self, audio_path: Path) -> None:
        if audio_path.suffix.lower() == ".mp3":
            self._play_mp3(audio_path)
            return

        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as exc:
            raise AudioIOError(
                "Audio dependencies are missing. Install project dependencies with `python -m pip install -e .`."
            ) from exc

        try:
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as exc:
            raise AudioIOError(f"Failed to play audio file: {audio_path}") from exc

    def _play_mp3(self, audio_path: Path) -> None:
        # Try native Windows MCI play via ctypes first (instantaneous, zero overhead)
        try:
            import ctypes
            winmm = ctypes.windll.winmm
            abs_path = str(audio_path.resolve())
            
            # Close first in case it was left open in a previous turn
            winmm.mciSendStringW("close mp3player", None, 0, 0)
            
            open_cmd = f'open "{abs_path}" type mpegvideo alias mp3player'
            res = winmm.mciSendStringW(open_cmd, None, 0, 0)
            if res != 0:
                raise RuntimeError(f"MCI open failed: {res}")
                
            res = winmm.mciSendStringW("play mp3player wait", None, 0, 0)
            if res != 0:
                raise RuntimeError(f"MCI play failed: {res}")
                
            winmm.mciSendStringW("close mp3player", None, 0, 0)
            return
        except Exception as exc:
            logger.warning("MCI player failed: %s. Falling back to PowerShell player.", exc)
            self._play_mp3_powershell(audio_path)

    def _play_mp3_powershell(self, audio_path: Path) -> None:
        player_script = (
            "Add-Type -AssemblyName presentationCore; "
            "$player = New-Object System.Windows.Media.MediaPlayer; "
            f"$player.Open([Uri]'{audio_path.as_uri()}'); "
            "while (-not $player.NaturalDuration.HasTimeSpan) "
            "{ Start-Sleep -Milliseconds 100 }; "
            "$player.Play(); "
            "while ($player.Position -lt $player.NaturalDuration.TimeSpan) "
            "{ Start-Sleep -Milliseconds 100 }; "
            "$player.Close()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", player_script],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            raise AudioIOError(f"Failed to play MP3 audio file: {audio_path}") from exc
