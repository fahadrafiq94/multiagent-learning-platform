import json
import logging
from typing import Any

import structlog

from app.config.settings import settings
from app.observability.events import (
    REQUEST_EVENT_LOGGER_NAME,
)


class _RequestEventJsonFormatter(logging.Formatter):
    """
    Renders one request-observability event as a single JSON line
    (4B.10 research export). Falls back to the plain message when a
    record carries no event payload.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        event = getattr(record, "event", None)

        if isinstance(event, dict):
            return json.dumps(event, ensure_ascii=False)

        return json.dumps({"message": record.getMessage()})


def configure_logging() -> None:
    """
    Configures logging for the application.
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # Render exc_info into the event dict so exceptions stay part
        # of the (single-line JSON) record instead of being appended
        # as unstructured multi-line text by the stdlib formatter.
        structlog.processors.format_exc_info,
    ]

    if settings.app_env == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Caching on first use permanently pins the processor chain
        # on the shared logger if anything logs before this function
        # runs, so it stays off to keep late configuration effective.
        cache_logger_on_first_use=False,
    )

    _attach_request_event_file_handler()


def _attach_request_event_file_handler() -> None:
    """
    Optional 4B.10 research export: when OBSERVABILITY_LOG_FILE is set,
    write each request event as one JSON line to that file. Propagation
    stays on, so the event still reaches the main application log too.
    """
    target = settings.observability_log_file

    if not target:
        return

    handler = logging.FileHandler(target, encoding="utf-8")

    handler.setFormatter(_RequestEventJsonFormatter())

    # Replace any previous handler so re-configuration never doubles
    # the lines written per event.
    for existing in logging.getLogger(REQUEST_EVENT_LOGGER_NAME).handlers[:]:
        if isinstance(existing, logging.FileHandler):
            existing.close()

            logging.getLogger(REQUEST_EVENT_LOGGER_NAME).removeHandler(existing)

    logging.getLogger(REQUEST_EVENT_LOGGER_NAME).addHandler(handler)
