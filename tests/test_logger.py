import logging

import hayhooks.server.logger as logger_module


class _CaptureSink:
    def __init__(self, calls: list[tuple]):
        self.calls = calls

    def log(self, *args):
        self.calls.append(args)


def test_intercept_handler_skips_excluded_uvicorn_access_logs(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(logger_module.log, "opt", lambda **_kwargs: _CaptureSink(calls))
    handler = logger_module._InterceptHandler(access_log_excluded_path_prefixes=["/dashboard/api/traces"])

    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("::1:12345", "GET", "/dashboard/api/traces?limit=10", "1.1", 200),
        exc_info=None,
    )

    handler.emit(record)

    assert calls == []


def test_intercept_handler_keeps_non_excluded_uvicorn_access_logs(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(logger_module.log, "opt", lambda **_kwargs: _CaptureSink(calls))
    handler = logger_module._InterceptHandler(access_log_excluded_path_prefixes=["/dashboard/api/traces"])

    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("::1:12345", "GET", "/status", "1.1", 200),
        exc_info=None,
    )

    handler.emit(record)

    assert len(calls) == 1
