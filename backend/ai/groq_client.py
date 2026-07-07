import json

from pydantic import BaseModel, ValidationError


class GroqClientError(RuntimeError):
    """Raised when a Groq structured completion fails."""


class GroqStructuredClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def complete_raw_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Return the raw parsed JSON dict without Pydantic validation."""
        if not self.api_key:
            raise GroqClientError("GROQ_API_KEY is required for Groq extraction mode.")
        try:
            completion = self._load_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = completion.choices[0].message.content
            if not content:
                raise GroqClientError("Groq returned an empty response.")
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise GroqClientError("Groq response was not valid JSON.") from exc
        except GroqClientError:
            raise
        except Exception as exc:
            raise GroqClientError("Groq raw JSON completion failed.") from exc

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        if not self.api_key:
            raise GroqClientError("GROQ_API_KEY is required for Groq extraction mode.")

        try:
            completion = self._load_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = completion.choices[0].message.content
            if not content:
                raise GroqClientError("Groq returned an empty response.")

            data = json.loads(content)
            return response_model.model_validate(data)
        except ValidationError as exc:
            raise GroqClientError("Groq response did not match the expected schema.") from exc
        except json.JSONDecodeError as exc:
            raise GroqClientError("Groq response was not valid JSON.") from exc
        except GroqClientError:
            raise
        except Exception as exc:
            raise GroqClientError("Groq structured completion failed.") from exc

    def _load_client(self):
        if self._client is not None:
            return self._client

        try:
            from groq import Groq
        except ImportError as exc:
            raise GroqClientError(
                "groq is missing. Install project dependencies with `python -m pip install -e .`."
            ) from exc

        self._client = Groq(api_key=self.api_key)
        return self._client

