"""
Tests for verifying that exception tracebacks are correctly logged using loguru.

These tests verify that `log.opt(exception=True)` properly includes exception
tracebacks in the log output, following the loguru migration guide pattern.

See: https://loguru.readthedocs.io/en/stable/resources/migration.html
"""

import pytest
from loguru import logger

from hayhooks.server.logger import log
from hayhooks.server.pipelines.utils import _process_pipeline_end


@pytest.fixture
def caplog_handler():
    """
    Fixture that captures loguru log records including exception info.

    Yields:
        tuple: (output_list, records_list) where output contains formatted messages
               and records contains the raw record dicts with exception info
    """
    output = []
    records = []

    def sink(message):
        # str(message) includes the formatted message + exception traceback if present
        output.append(str(message))
        records.append(message.record)

    # Using format="{message}" - loguru appends exception traceback automatically
    handler_id = logger.add(sink, format="{message}", level="DEBUG")
    yield output, records
    logger.remove(handler_id)


def test_log_opt_exception_captures_traceback(caplog_handler):
    output, records = caplog_handler

    try:
        raise ValueError("Test exception")  # noqa: EM101
    except ValueError:
        log.opt(exception=True).error("An error occurred")

    assert len(records) == 1
    record = records[0]

    # Verify exception is captured in record
    assert record["exception"] is not None
    assert record["exception"].type is ValueError
    assert record["exception"].traceback is not None

    # Verify traceback is printed in log output
    log_output = output[0]
    assert "Traceback" in log_output
    assert "ValueError" in log_output
    assert "Test exception" in log_output


def test_process_pipeline_end_logs_exception_with_traceback(caplog_handler):
    output, records = caplog_handler

    def failing_callback(result):
        raise ValueError("Pipeline callback error")  # noqa: EM101

    result = _process_pipeline_end({"test": "data"}, failing_callback)

    assert result is None
    assert len(records) == 1

    # Verify the log message
    log_output = output[0]
    assert "Error in on_pipeline_end callback" in log_output

    # Verify traceback is printed in log output
    assert "Traceback" in log_output
    assert "ValueError" in log_output
    assert "Pipeline callback error" in log_output

    # Verify exception is captured in record
    assert records[0]["exception"] is not None
    assert records[0]["exception"].type is ValueError
