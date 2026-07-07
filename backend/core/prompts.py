from pathlib import Path


class PromptLoadError(RuntimeError):
    """Raised when a prompt file cannot be loaded."""


class PromptLoader:
    def __init__(self, prompt_dir: Path | None = None) -> None:
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts"

    def load(self, name: str) -> str:
        prompt_path = self.prompt_dir / name
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise PromptLoadError(f"Prompt file not found: {prompt_path}") from exc

