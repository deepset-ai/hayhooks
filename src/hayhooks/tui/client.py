"""GET-only dashboard connection, also usable without the TUI extra."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import os
import re
import unicodedata
from http import HTTPStatus
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests


class DashboardError(RuntimeError):
    """A safe display message, without URLs, credentials or response bodies."""


def literal(value: object, limit: int = 8192) -> str:
    """Neutralize ANSI, OSC, C0/C1 and bidi controls before literal Rich rendering."""
    text = str(value)[:limit]
    text = re.sub(r"(?:\x1b\]|\x9d)[^\x07\x1b]*(?:\x07|\x1b\\)?", "", text)
    text = re.sub(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]", "", text)
    return "".join(char for char in text if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf"})


class DashboardClient:
    def __init__(self, url: str, *, timeout: tuple[float, float] = (3.0, 5.0)) -> None:
        try:
            parts = urlsplit(url)
        except ValueError as error:
            raise DashboardError("Invalid server URL") from error
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise DashboardError("Use an HTTP(S) URL without credentials, query or fragment")
        self.url = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
        self.label = literal(parts.netloc + parts.path.rstrip("/"), 200)
        self.timeout = timeout
        self.session = requests.Session()
        if token := os.environ.get("HAYHOOKS_DASHBOARD_TOKEN"):
            self.session.headers["Authorization"] = f"Bearer {token}"
        if key := os.environ.get("HAYHOOKS_DASHBOARD_API_KEY"):
            header = os.environ.get("HAYHOOKS_DASHBOARD_API_KEY_HEADER", "X-API-Key")
            if not re.fullmatch(r"[A-Za-z0-9-]+", header):
                self.session.close()
                raise DashboardError("Invalid API key header name")
            self.session.headers[header] = key

    def get(self, path: str, **params: Any) -> dict[str, Any]:  # noqa: C901, PLR0912
        try:
            with self.session.get(
                self.url + "/" + path.lstrip("/"),
                params=params,
                timeout=self.timeout,
                verify=True,
                allow_redirects=False,
            ) as response:
                if response.status_code in {401, 403}:
                    raise DashboardError("Authentication failed or monitoring access denied (401/403)")
                if response.status_code == HTTPStatus.NOT_FOUND:
                    if path == "status":
                        raise DashboardError("Status API unavailable (404)")
                    if path.endswith("/executions"):
                        raise DashboardError("Execution enumeration unavailable or monitoring disabled (404)")
                    raise DashboardError("Execution expired or no longer accessible (404)")
                if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
                    message = "Server unavailable"
                    try:
                        detail = response.json().get("detail")
                    except (ValueError, AttributeError):
                        detail = None
                    if detail == "Durable execution store is unavailable":
                        message = "Execution store unavailable"
                    elif detail == "Durable execution state is invalid":
                        message = "Execution metadata is invalid"
                    raise DashboardError(message)
                if response.status_code != HTTPStatus.OK:
                    raise DashboardError(f"Request failed (HTTP {response.status_code})")
                data = response.json()
            if not isinstance(data, dict):
                raise DashboardError("Server returned an invalid snapshot")
            return data
        except requests.Timeout as error:
            raise DashboardError("Connection timed out") from error
        except requests.RequestException as error:
            raise DashboardError("Connection unavailable") from error
        except ValueError as error:
            raise DashboardError("Server returned invalid JSON") from error

    def status(self) -> dict[str, Any]:
        data = self.get("status")
        if (
            not isinstance(data.get("pipelines"), list)
            or not isinstance(data.get("status"), str)
            or not all(isinstance(name, str) for name in data["pipelines"])
        ):
            raise DashboardError("Server returned an invalid status snapshot")
        if "deployments" in data and (
            not isinstance(data["deployments"], list)
            or not all(
                isinstance(row, dict)
                and all(isinstance(row.get(key), str) for key in ("name", "type", "execution_mode", "state"))
                and (row.get("health") is None or isinstance(row["health"], dict))
                for row in data["deployments"]
            )
        ):
            raise DashboardError("Server returned invalid deployment summaries")
        return data

    def executions(self, pipeline: str, **params: Any) -> dict[str, Any]:
        data = self.get(f"{quote(pipeline, safe='')}/executions", **{k: v for k, v in params.items() if v is not None})
        if (
            not isinstance(data.get("executions"), list)
            or not isinstance(data.get("has_more"), bool)
            or (data.get("next_cursor") is not None and not isinstance(data["next_cursor"], str))
            or (data.get("has_more") and not data.get("next_cursor"))
            or not all(
                isinstance(row, dict)
                and all(isinstance(row.get(key), str) for key in ("execution_id", "status", "updated_at"))
                for row in data["executions"]
            )
        ):
            raise DashboardError("Server returned an invalid execution page")
        return data

    def detail(self, pipeline: str, execution_id: str) -> dict[str, Any]:
        data = self.get(f"{quote(pipeline, safe='')}/executions/{quote(execution_id, safe='')}")
        data.pop("result", None)
        data.pop("links", None)
        if data.get("execution_id") != execution_id:
            raise DashboardError("Server returned details for a different execution")
        return data

    def close(self) -> None:
        self.session.close()
