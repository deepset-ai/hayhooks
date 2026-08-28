"""Run concurrent durable REST/SSE executions using only the Python standard library."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


def _http_url(base_url: str, path: str) -> str:
    url = urljoin(base_url, path)
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {url}")
    return url


def run_execution(base_url: str, index: int) -> None:
    request = Request(  # noqa: S310 - URL scheme validated by _http_url
        _http_url(base_url, "/durable_execution/run-durable"),
        data=json.dumps({"value": index, "require_approval": False, "fail_once": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - URL scheme validated above
        stream_url = _http_url(base_url, json.load(response)["links"]["stream"])
    event = data = None
    with urlopen(stream_url, timeout=60) as response:  # noqa: S310 - URL scheme validated above
        for raw_line in response:
            line = raw_line.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data = line.partition(":")[2].strip()
            elif not line and event:
                print(f"[{index}] {event}: {data}")
                if event in {"completed", "failed", "canceled"}:
                    return
                event = data = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:1416")
    parser.add_argument("--executions", type=int, default=3)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.executions) as pool:
        tuple(pool.map(partial(run_execution, args.base_url), range(1, args.executions + 1)))


if __name__ == "__main__":
    main()
