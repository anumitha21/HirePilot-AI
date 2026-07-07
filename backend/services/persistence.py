from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class InterviewStore:
    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path or "backend/data/interviews.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = self.list()
        records.append(payload)
        self.storage_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return payload

    def list(self) -> list[dict[str, Any]]:
        try:
            data = self.storage_path.read_text(encoding="utf-8")
            return json.loads(data) if data.strip() else []
        except json.JSONDecodeError:
            return []

    def get_by_candidate(self, candidate_name: str) -> list[dict[str, Any]]:
        return [record for record in self.list() if record.get("candidate_name") == candidate_name]
