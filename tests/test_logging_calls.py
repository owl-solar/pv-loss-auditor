"""Logging calls must be valid: a bad keyword only fails at run time, on the rarely taken path.

Found by mypy after the print -> logging conversion: `print(..., flush=True)` had become
`log.info(..., flush=True)`, which raises TypeError the first time a download retries.
"""

import ast
from pathlib import Path

import pytest

from pv_loss_auditor import pvdaq

PACKAGE = Path("src/pv_loss_auditor")
LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}
ALLOWED_KEYWORDS = {
    "exc_info",
    "stack_info",
    "stacklevel",
    "extra",
}  # the only keywords Logger methods accept


def logging_calls():
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LEVELS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "log"
            ):
                yield path.name, node


def test_logging_calls_use_only_keywords_that_logger_accepts():
    bad = [
        (name, node.lineno, kw.arg)
        for name, node in logging_calls()
        for kw in node.keywords
        if kw.arg not in ALLOWED_KEYWORDS
    ]
    assert bad == []


def test_there_are_logging_calls_to_check():
    assert sum(1 for _ in logging_calls()) > 20


def test_download_retry_logs_and_succeeds_after_a_dropped_connection(monkeypatch, caplog):
    # A mobile connection drops once: the retry path logs a message, waits, and then returns the bytes.
    calls = {"n": 0}

    def flaky(s3, key, start, end):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection reset")
        return b"x" * (end - start)

    monkeypatch.setattr(pvdaq, "_get_range", flaky)
    monkeypatch.setattr(pvdaq.time, "sleep", lambda s: None)
    with caplog.at_level("INFO"):
        body = pvdaq._get_range_retry(None, "key", 0, 10, retries=2)
    assert body == b"x" * 10
    assert "network error (OSError)" in caplog.text


def test_download_retry_gives_up_after_the_last_attempt(monkeypatch):
    def always_fails(s3, key, start, end):
        raise OSError("connection reset")

    monkeypatch.setattr(pvdaq, "_get_range", always_fails)
    monkeypatch.setattr(pvdaq.time, "sleep", lambda s: None)
    with pytest.raises(OSError, match="connection reset"):
        pvdaq._get_range_retry(None, "key", 0, 10, retries=1)
