"""Present exact Gemma3 documents and retain an explicit, local acceptance record."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .storage import now, read_json, sha, write_json


GEMMA3_PROFILE_ID = "gemma3-4b"
TERMS_VERSION = "2026-04-01"
DOCUMENTS = (
    {"path": "licenses/models/gemma3-terms.txt", "title": "Gemma Terms of Use",
     "url": "https://ai.google.dev/gemma/terms", "version": "2026-04-01",
     "sha256": "eb1dd936eee8cf0f3c84573a823607f2ec1e89f1e0ab7d6553bebd95407e0139"},
    {"path": "licenses/models/gemma3-prohibited-use-policy.txt", "title": "Gemma Prohibited Use Policy",
     "url": "https://ai.google.dev/gemma/prohibited_use_policy", "version": "2024-02-21",
     "sha256": "32e4032bb8934ee14d1c61157d42fd307ee70c4a7ae0b9781e7dbae1288886ef"},
)


class ModelTermsError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ModelTerms:
    """No operation implicitly accepts terms; only an explicit GUI action records it."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.path = self.root / "data/model-terms.json"

    def _documents(self):
        documents = []
        for definition in DOCUMENTS:
            path = self.root / definition["path"]
            try:
                if not path.is_file() or sha(path) != definition["sha256"]:
                    raise ValueError("terms file identity mismatch")
                content = path.read_text(encoding="utf-8-sig")
                if not content.strip():
                    raise ValueError("empty terms")
            except (OSError, ValueError, UnicodeError) as exc:
                raise ModelTermsError("TERMS_UNAVAILABLE",
                    "Gemma3の利用条件ファイルが不足または破損しています。公開ZIPから復元してください。") from exc
            documents.append({**definition, "text": content})
        return documents

    def _registry(self):
        if not self.path.exists():
            return {"version": 1, "models": {}}
        try:
            registry = read_json(self.path)
            if (not isinstance(registry, dict) or registry.get("version") != 1 or
                    not isinstance(registry.get("models"), dict)):
                raise ValueError("invalid terms registry")
            for profile_id, record in registry["models"].items():
                if (profile_id != GEMMA3_PROFILE_ID or not isinstance(record, dict) or
                        record.get("explicit_acceptance") is not True or
                        record.get("acceptance_source") != "interactive-gui" or
                        not isinstance(record.get("accepted_at"), str) or
                        not isinstance(record.get("terms_version"), str) or
                        not isinstance(record.get("documents"), list)):
                    raise ValueError("invalid acceptance record")
            return registry
        except (OSError, ValueError, TypeError) as exc:
            raise ModelTermsError("INVALID_TERMS_RECORD",
                "利用条件への同意記録を読めません。自動で同意済みに変更しません。") from exc

    def status(self, profile_id: str):
        if profile_id != GEMMA3_PROFILE_ID:
            return {"required": False, "accepted": True, "documents": []}
        documents = self._documents()
        identities = [{key: value for key, value in document.items() if key != "text"}
                      for document in documents]
        record = self._registry()["models"].get(profile_id)
        accepted = bool(record and record.get("terms_version") == TERMS_VERSION and
                        record.get("documents") == identities)
        return {"required": True, "accepted": accepted, "terms_version": TERMS_VERSION,
                "documents": documents, "acceptance": deepcopy(record) if accepted else None}

    def accept(self, profile_id: str, document_hashes: dict[str, str], *, accepted: bool):
        # A caller must pass the hashes of the documents the user actually saw.
        # The GUI calls this only from its unchecked-by-default consent dialog.
        if profile_id != GEMMA3_PROFILE_ID or accepted is not True:
            raise ModelTermsError("TERMS_NOT_ACCEPTED", "Gemma3の利用条件への明示的な同意が必要です。")
        status = self.status(profile_id)
        documents = status["documents"]
        if document_hashes != {item["path"]: item["sha256"] for item in documents}:
            raise ModelTermsError("TERMS_CHANGED", "表示後に利用条件が変更されました。もう一度確認してください。")
        registry = self._registry()
        record = {"terms_version": TERMS_VERSION,
                  "documents": [{key: value for key, value in item.items() if key != "text"}
                                for item in documents],
                  "explicit_acceptance": True, "acceptance_source": "interactive-gui", "accepted_at": now()}
        registry["models"][profile_id] = record
        write_json(self.path, registry)
        return deepcopy(record)

    def require(self, profile_id: str):
        status = self.status(profile_id)
        if not status["accepted"]:
            raise ModelTermsError("TERMS_NOT_ACCEPTED",
                "モデル管理画面でGemma3の利用規約・禁止用途を読み、同意してから実行してください。")
        return status.get("acceptance")


def require_model_terms(root: Path, profile_id: str):
    return ModelTerms(root).require(profile_id)
