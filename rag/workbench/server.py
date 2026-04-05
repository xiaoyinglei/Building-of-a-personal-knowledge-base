from __future__ import annotations

import json
import socket
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel

from rag.workbench.service import WorkbenchService

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_workbench_server(
    *,
    storage_root: Path,
    workspace_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    service = WorkbenchService(storage_root=storage_root, workspace_root=workspace_root)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                payload = service.get_state(
                    active_profile_id=_single(query, "profile_id"),
                    sync=_single(query, "sync") != "0",
                )
                self._write_json(payload)
                return
            if parsed.path in {"/", "/index.html"}:
                self._write_static("index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/app.css":
                self._write_static("app.css", "text/css; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._write_static("app.js", "application/javascript; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = self._read_json()
                if parsed.path == "/api/query":
                    query_payload = service.query(
                        query_text=_require_text(body, "query"),
                        mode=str(body.get("mode", "mix")),
                        profile_id=_optional_text(body, "profile_id"),
                        source_scope=_string_list(body.get("source_scope")),
                    )
                    self._write_json(query_payload)
                    return
                if parsed.path == "/api/files/save":
                    operation_payload = service.save_file(
                        relative_path=_require_text(body, "relative_path"),
                        profile_id=_optional_text(body, "profile_id"),
                        content_text=_optional_text(body, "content_text"),
                        content_base64=_optional_text(body, "content_base64"),
                        auto_ingest=bool(body.get("auto_ingest", True)),
                    )
                    self._write_json(operation_payload)
                    return
                if parsed.path == "/api/files/ingest":
                    operation_payload = service.ingest_file(
                        relative_path=_require_text(body, "relative_path"),
                        profile_id=_optional_text(body, "profile_id"),
                    )
                    self._write_json(operation_payload)
                    return
                if parsed.path == "/api/files/rebuild":
                    operation_payload = service.rebuild_file(
                        relative_path=_require_text(body, "relative_path"),
                        profile_id=_optional_text(body, "profile_id"),
                    )
                    self._write_json(operation_payload)
                    return
                if parsed.path == "/api/files/delete":
                    operation_payload = service.delete_file(
                        relative_path=_require_text(body, "relative_path"),
                        profile_id=_optional_text(body, "profile_id"),
                    )
                    self._write_json(operation_payload)
                    return
                if parsed.path == "/api/sync":
                    state_payload = service.get_state(active_profile_id=_optional_text(body, "profile_id"), sync=True)
                    self._write_json(state_payload)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self._write_json(
                    {"ok": False, "message": "Request failed", "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise RuntimeError("Expected a JSON object body")
            return data

        def _write_static(self, name: str, content_type: str) -> None:
            payload = (_STATIC_DIR / name).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _write_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)


def run_workbench_server(
    *,
    storage_root: Path,
    workspace_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> str:
    server = create_workbench_server(
        storage_root=storage_root,
        workspace_root=workspace_root,
        host=host,
        port=port,
    )
    bound_host, bound_port = cast("tuple[str, int]", server.server_address)
    url_host = host if host not in {"0.0.0.0", ""} else "localhost"
    url = f"http://{url_host}:{bound_port}"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(url, flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return url


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _jsonable(payload: object) -> object:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, list):
        return [_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    return payload


def _single(values: dict[str, list[str]], key: str) -> str | None:
    matches = values.get(key)
    if not matches:
        return None
    value = matches[0].strip()
    return value or None


def _optional_text(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _require_text(payload: dict[str, object], key: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise RuntimeError(f"Missing required field: {key}")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


__all__ = ["create_workbench_server", "find_free_port", "run_workbench_server"]
