from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from app.config.settings import settings
from app.logging.config import configure_logging
from app.observability.events import (
    REQUEST_EVENT_LOGGER_NAME,
)


@pytest.fixture
def request_event_logger() -> Iterator[logging.Logger]:
    """
    Hands out the request-events logger with any file handlers
    added by this test removed again afterwards.
    """
    logger = logging.getLogger(REQUEST_EVENT_LOGGER_NAME)

    snapshot = list(logger.handlers)

    yield logger

    for handler in logger.handlers[:]:
        if handler not in snapshot:
            handler.close()

            logger.removeHandler(handler)


def _emit_sample_event() -> None:
    logging.getLogger(REQUEST_EVENT_LOGGER_NAME).info(
        "request_completed",
        extra={
            "event": {
                "trace_id": None,
                "outcome": ("completed"),
                "route": None,
                "duration_ms": (1.5),
            }
        },
    )


def test_no_file_handler_when_observability_log_file_is_unset(
    request_event_logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "observability_log_file", None)

    configure_logging()

    assert not any(
        isinstance(handler, logging.FileHandler) for handler in request_event_logger.handlers
    )


def test_file_handler_writes_one_json_line_per_event_and_propagates(
    request_event_logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "request-events.jsonl"

    monkeypatch.setattr(
        settings,
        "observability_log_file",
        str(target),
    )

    configure_logging()

    with caplog.at_level(
        logging.INFO,
        logger=(REQUEST_EVENT_LOGGER_NAME),
    ):
        _emit_sample_event()

        _emit_sample_event()

    lines = target.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2

    for line in lines:
        payload = json.loads(line)

        assert payload["outcome"] == "completed"

        assert payload["duration_ms"] == 1.5

    assert any(record.name == REQUEST_EVENT_LOGGER_NAME for record in caplog.records)


def test_reconfigure_replaces_existing_file_handler(
    request_event_logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = tmp_path / "first.jsonl"

    second = tmp_path / "second.jsonl"

    monkeypatch.setattr(
        settings,
        "observability_log_file",
        str(first),
    )

    configure_logging()

    with caplog.at_level(
        logging.INFO,
        logger=(REQUEST_EVENT_LOGGER_NAME),
    ):
        _emit_sample_event()

    monkeypatch.setattr(
        settings,
        "observability_log_file",
        str(second),
    )

    configure_logging()

    with caplog.at_level(
        logging.INFO,
        logger=(REQUEST_EVENT_LOGGER_NAME),
    ):
        _emit_sample_event()

    assert first.read_text(encoding="utf-8").strip().count("\n") == 0

    assert len(first.read_text(encoding="utf-8").strip().splitlines()) == 1

    assert len(second.read_text(encoding="utf-8").strip().splitlines()) == 1
