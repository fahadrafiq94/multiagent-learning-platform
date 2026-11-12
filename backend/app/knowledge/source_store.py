from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import Field

from app.knowledge.errors import KnowledgeIntegrityError, KnowledgeNotFound
from app.knowledge.schemas import Digest, KnowledgeModel


class SourceReference(KnowledgeModel):
    source_hash: Digest
    size_bytes: int = Field(ge=1)

    @property
    def uri(self) -> str:
        return f"sha256:{self.source_hash}"


class SourceStore(Protocol):
    """Storage is opaque; callers authorize document access through repositories first."""

    def put(self, content: bytes) -> SourceReference: ...

    def get(self, source_hash: str) -> bytes: ...


class FileSourceStore:
    def __init__(self, root: Path, *, max_bytes: int = 25 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("Source size limit must be positive.")
        self.root = root.resolve()
        self.max_bytes = max_bytes

    def _path(self, digest: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("A SHA-256 source identifier is required.")
        path = self.root / digest[:2] / digest
        if not path.resolve().is_relative_to(self.root):
            raise KnowledgeIntegrityError("Source path escapes the managed root.")
        return path

    def put(self, content: bytes) -> SourceReference:
        if not content or len(content) > self.max_bytes:
            raise ValueError("Source is empty or exceeds its configured size limit.")
        digest = hashlib.sha256(content).hexdigest()
        path = self._path(digest)
        if path.exists():
            if self.get(digest) != content:
                raise KnowledgeIntegrityError("Existing source is corrupt.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".source-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        return SourceReference(source_hash=digest, size_bytes=len(content))

    def get(self, source_hash: str) -> bytes:
        path = self._path(source_hash)
        try:
            if path.stat().st_size > self.max_bytes:
                raise KnowledgeIntegrityError("Stored source exceeds the configured size limit.")
            content = path.read_bytes()
        except FileNotFoundError:
            raise KnowledgeNotFound("Source not found.") from None
        if hashlib.sha256(content).hexdigest() != source_hash:
            raise KnowledgeIntegrityError("Stored source hash does not match its identifier.")
        return content
