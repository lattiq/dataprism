"""HTTP server that serves the EDA results dashboard."""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from importlib import resources
from pathlib import Path
from typing import Union


def _find_free_port() -> int:
    """Find an available port using the OS socket trick."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _build_html(json_data: dict) -> str:
    """Read template.html and inject JSON data into the placeholder."""
    template = resources.files("dataprism.viewer").joinpath("template.html").read_text("utf-8")
    if "__EDA_DATA_PLACEHOLDER__" not in template:
        raise RuntimeError("Template is missing __EDA_DATA_PLACEHOLDER__. Cannot inject data.")
    minified = json.dumps(json_data, separators=(",", ":"))
    # Escape </script> sequences to prevent premature script tag closure
    minified = minified.replace("</", "<\\/")
    return template.replace("__EDA_DATA_PLACEHOLDER__", minified)


class _SinglePageHandler(SimpleHTTPRequestHandler):
    """Serves a single HTML page from memory."""

    html_bytes: bytes = b""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.html_bytes)))
        self.end_headers()
        self.wfile.write(self.html_bytes)

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # silence request logs


DEFAULT_PORT = 8765


def serve_results(
    data: Union[str, Path, dict],
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """Serve EDA results as an interactive dashboard in the browser.

    Parameters
    ----------
    data : str, Path, or dict
        Path to a JSON file, or an already-loaded dict of EDA results.
    port : int
        Port to serve on (default: 8765).
    open_browser : bool
        Whether to open the default browser automatically.
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        with open(path, encoding="utf-8") as f:
            json_data = json.load(f)
    else:
        json_data = data

    if port == 0:
        port = _find_free_port()


    html = _build_html(json_data)
    html_bytes = html.encode("utf-8")

    class _Handler(_SinglePageHandler):
        pass

    _Handler.html_bytes = html_bytes

    server = HTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}"

    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()

    print(f"Serving EDA dashboard at {url}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
