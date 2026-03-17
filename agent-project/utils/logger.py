"""Structured logger with trace id support."""

import logging
import sys
import uuid


def get_logger():
    logger = logging.getLogger("agent")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s extra=%(extra)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def new_trace_id() -> str:
    return uuid.uuid4().hex


# convenience to attach to logger instance
logging.Logger.new_trace_id = staticmethod(new_trace_id)  # type: ignore[attr-defined]
