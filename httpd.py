#!/usr/bin/env python3
# httpd.py - ASS_HTTPd v3.0: Quantum-Sentient HTTP Server for QyrinthOS
#
# - Async/await native (no nested asyncio.run)
# - Single logger (no duplicate log lines)
# - Clean SSL handling (HTTP -> HTTPS fallback)
# - Compatible with main.py: await run_quantum_server(quantum_core)
# - Uses local QUANTUM_CORE helper instances (trainer, LASER, etc.)

import asyncio
import ssl
import json
import time
import hashlib
import hmac
import secrets
import logging
import random
import re
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import base64
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (single handler, no duplicate lines)
# ---------------------------------------------------------------------------

logger = logging.getLogger("QuantumHTTPd")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - QuantumHTTPd - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Do not propagate to root logger to avoid duplicate output
logger.propagate = False

# ---------------------------------------------------------------------------
# QYRINTHOS MODULE IMPORTS (robust, stub-friendly)
# ---------------------------------------------------------------------------

# Allow importing local QyrinthOS modules
sys.path.append(os.path.dirname(__file__))

QYRINTH_MODULES_LOADED = False

try:
    from bugginrace import MilitaryGradeEvolutionaryTrainer as EvolutionaryEcosystem
    from bumpy import BumpyArray as SentientArray
    from laser import LASERUtility
    from qubitlearn import QubitLearn
    # qualia_ritual lives in sentiflow, not laser
    from sentiflow import SentientTensor, nn, optim, qualia_ritual

    QYRINTH_MODULES_LOADED = True
    logger.info("QyrinthOS HTTPd: Core quantum modules imported successfully.")
except Exception as e:
    logger.warning(
        "QyrinthOS modules not fully available. Running HTTP server in stub mode: %s", e
    )

    # --- Minimal stubs so the server can still run ---

    class EvolutionaryEcosystem:  # type: ignore[override]
        def __init__(self, *_, **__):
            logger.info("Stub EvolutionaryEcosystem initialized.")

        def evolutionary_race_cycle(self, steps: int) -> None:
            logger.info("Stub evolutionary_race_cycle(%d) called.", steps)

    class SentientArray:  # type: ignore[override]
        def __init__(self, *_, **__):
            self.data = []

    class LASERUtility:  # type: ignore[override]
        def __init__(self, *_, **__):
            logger.info("Stub LASERUtility initialized.")

        def log_event(self, *_, **__):
            pass

        def set_coherence_level(self, *_, **__):
            pass

    class QubitLearn:  # type: ignore[override]
        def __init__(self, *_, **__):
            logger.info("Stub QubitLearn initialized.")

        def trigger_cognitive_leap(self) -> None:
            logger.info("Stub cognitive leap triggered.")

    class SentientTensor:  # type: ignore[override]
        def __init__(self, *_, **__):
            self.data = []

    class nn:  # type: ignore[override]
        class Dense:
            def __init__(self, *_, **__):
                pass

            def __call__(self, x):
                return x

        class TransdimensionalConv:
            def __init__(self, *_, **__):
                pass

            def __call__(self, x):
                return x

    class optim:  # type: ignore[override]
        class Adam:
            def __init__(self, *_, **__):
                pass

            def step(self):
                pass

    def qualia_ritual(*_, **__) -> None:  # type: ignore[override]
        return None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8443

SECRET_KEY = secrets.token_bytes(32)
SESSION_EXPIRY_MINUTES = 60

CONTEXT_CERT = "host.cert"
CONTEXT_KEY = "host.key"

# Local to the HTTP daemon. When QYRINTH modules are available it will hold
# helper instances for evolution and logging.
QUANTUM_CORE: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Cookie + Session helpers
# ---------------------------------------------------------------------------


def generate_session_id() -> str:
    return secrets.token_urlsafe(32)


def sign_data(data: str) -> str:
    digest = hmac.new(SECRET_KEY, data.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def check_signature(data: str, signature: str) -> bool:
    return hmac.compare_digest(sign_data(data), signature)


def create_secure_cookie(
    name: str, value: str, expires_in_minutes: int = SESSION_EXPIRY_MINUTES
) -> str:
    expiry_time = datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)
    expires_str = expiry_time.strftime("%a, %d-%b-%Y %H:%M:%S GMT")

    signed_value = f"{value}|{sign_data(value)}"

    cookie_str = (
        f"{name}={signed_value}; "
        f"Expires={expires_str}; "
        f"Path=/; "
        f"Secure; "
        f"HttpOnly; "
        f"SameSite=Strict"
    )
    return cookie_str


def _parse_cookies(cookie_header: str) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    if not cookie_header:
        return cookies

    # Very simple cookie parser: key=value; key2=value2
    pairs = re.findall(r"(\S+?)=(\S+?)(?:;|$)", cookie_header)
    for name, value in pairs:
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        cookies[name] = value
    return cookies


def _encode_session(session: Dict[str, Any]) -> str:
    raw = json.dumps(session, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _decode_session(encoded: str) -> Dict[str, Any]:
    try:
        padding = "=" * (-len(encoded) % 4)
        raw = base64.urlsafe_b64decode(encoded + padding)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# HTTP Request handler
# ---------------------------------------------------------------------------


class QuantumHTTPRequestHandler:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.headers: Dict[str, str] = {}
        self.path: str = ""
        self.method: str = ""
        self.version: str = ""
        self.cookies: Dict[str, str] = {}
        self.session: Dict[str, Any] = {}

    async def _read_headers(self) -> bool:
        try:
            request_line = await asyncio.wait_for(self.reader.readline(), timeout=5)
            if not request_line:
                return False

            parts = request_line.decode("utf-8", errors="ignore").strip().split()
            if len(parts) != 3:
                return False

            self.method, full_path, self.version = parts
            self.path = full_path.split("?", 1)[0]

            while True:
                line = await asyncio.wait_for(self.reader.readline(), timeout=5)
                if not line or line == b"\r\n":
                    break
                try:
                    decoded = line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError:
                    continue
                if ":" in decoded:
                    key, value = decoded.split(":", 1)
                    self.headers[key.strip()] = value.strip()

            return True
        except asyncio.TimeoutError:
            logger.warning("Request timed out while reading headers.")
            return False
        except Exception as e:
            logger.error("Error while reading request: %s", e, exc_info=True)
            return False

    async def _load_session(self) -> None:
        cookie_header = self.headers.get("Cookie", "")
        self.cookies = _parse_cookies(cookie_header)

        raw_cookie = self.cookies.get("qyrinthos_session")
        if not raw_cookie:
            return

        try:
            value, signature = raw_cookie.split("|", 1)
        except ValueError:
            logger.warning("Malformed session cookie received.")
            return

        if not check_signature(value, signature):
            logger.warning("Invalid session signature detected.")
            return

        session = _decode_session(value)
        if session:
            session["is_new"] = False
            self.session = session
            logger.info("Session restored for user %s", self.session.get("user_id"))

    async def _ensure_session(self) -> None:
        if self.session:
            return

        new_id = generate_session_id()
        self.session = {
            "session_id": new_id,
            "user_id": secrets.token_urlsafe(16),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_new": True,
        }
        logger.info("New session created: %s", new_id)

    def _session_cookie_header(self) -> str:
        encoded = _encode_session(self.session)
        return create_secure_cookie("qyrinthos_session", encoded)

    async def _send_response(
        self,
        status_code: int,
        status_message: str,
        body: str,
        content_type: str = "text/html",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not isinstance(body, (str, bytes)):
            body = str(body)
        if isinstance(body, str):
            body_bytes = body.encode("utf-8")
        else:
            body_bytes = body

        response_lines = [
            f"HTTP/1.1 {status_code} {status_message}\r\n",
            f"Content-Type: {content_type}\r\n",
            f"Content-Length: {len(body_bytes)}\r\n",
            "Connection: close\r\n",
        ]

        if self.session and (
            self.session.get("is_new") or "qyrinthos_session" not in self.cookies
        ):
            response_lines.append(f"Set-Cookie: {self._session_cookie_header()}\r\n")

        if extra_headers:
            for k, v in extra_headers.items():
                response_lines.append(f"{k}: {v}\r\n")

        response_lines.append("\r\n")
        header_bytes = "".join(response_lines).encode("utf-8")

        self.writer.write(header_bytes + body_bytes)
        await self.writer.drain()

    async def handle_request(self) -> None:
        try:
            ok = await self._read_headers()
            if not ok:
                return

            await self._load_session()
            await self._ensure_session()

            logger.info(
                "Handling %s %s from user %s",
                self.method,
                self.path,
                self.session.get("user_id", "unknown"),
            )

            if self.path == "/":
                await self._serve_dashboard()
            elif self.path == "/api/status":
                await self._serve_api_status()
            elif self.path == "/api/evolution/run" and self.method.upper() == "POST":
                await self._handle_evolution_run()
            elif self.path == "/favicon.ico":
                await self._send_response(
                    404,
                    "Not Found",
                    "<h1>404 Not Found</h1>",
                )
            else:
                await self._send_response(
                    404,
                    "Not Found",
                    "<h1>404 Not Found</h1>",
                )
        except Exception as e:
            logger.error("Error handling request: %s", e, exc_info=True)
            await self._send_response(
                500,
                "Internal Server Error",
                "<h1>500 Internal Server Error</h1>",
            )
        finally:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass

    async def _serve_dashboard(self) -> None:
        quantum_loaded = "Yes" if QYRINTH_MODULES_LOADED else "No (Stub Mode)"

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>QyrinthOS Quantum Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; background:#0d1117; color:#c9d1d9; }}
    .container {{ max-width: 900px; margin: 40px auto; padding: 24px; border:1px solid #30363d; border-radius:8px; background:#010409; }}
    h1 {{ color:#58a6ff; margin-top:0; }}
    pre {{ background:#161b22; padding:10px; border-radius:4px; overflow-x:auto; }}
    button {{ background:#238636; color:#fff; padding:10px 18px; border:none; border-radius:6px; cursor:pointer; margin-top:8px; }}
    button:hover {{ background:#2ea043; }}
    .metric {{ padding:6px 0; border-bottom:1px solid #21262d; font-size:14px; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#161b22; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>QyrinthOS Quantum Dashboard</h1>
    <p>Welcome, User ID: <strong>{self.session.get("user_id", "Unknown")}</strong></p>
    <div class="metric">Session ID: <span class="pill">{self.session.get("session_id", "N/A")}</span></div>
    <div class="metric">Quantum Modules Loaded: <span class="pill">{quantum_loaded}</span></div>

    <h2>Evolutionary Core</h2>
    <p>Trigger an evolutionary consciousness emergence cycle:</p>
    <button onclick="runEvolution()">Start Evolutionary Race</button>
    <div id="status" style="margin-top:16px;"></div>

    <h2>API Probe</h2>
    <p><code>GET /api/status</code> to inspect quantum server state.</p>
  </div>
  <script>
    async function runEvolution() {{
      const statusDiv = document.getElementById('status');
      statusDiv.innerHTML = '<p style="color:#d29922;">Initiating Quantum Race...</p>';
      try {{
        const response = await fetch('/api/evolution/run', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ steps: 100 }})
        }});
        const result = await response.json();
        if (response.ok) {{
          statusDiv.innerHTML = '<p style="color:#58a6ff;">Evolutionary Cycle Complete!</p>' +
            '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
        }} else {{
          statusDiv.innerHTML = '<p style="color:#f85149;">Error: ' + (result.error || 'Unknown') + '</p>';
        }}
      }} catch (err) {{
        statusDiv.innerHTML = '<p style="color:#f85149;">Network Error: ' + err.message + '</p>';
      }}
    }}
  </script>
</body>
</html>
"""
        await self._send_response(200, "OK", html, "text/html; charset=utf-8")

    async def _serve_api_status(self) -> None:
        status_payload = {
            "status": "Quantum Active",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modules_loaded": QYRINTH_MODULES_LOADED,
            "session_id": self.session.get("session_id"),
            "user_id": self.session.get("user_id"),
            "quantum_core_keys": sorted(list(QUANTUM_CORE.keys())),
        }
        await self._send_response(
            200,
            "OK",
            json.dumps(status_payload),
            content_type="application/json",
        )

    async def _handle_evolution_run(self) -> None:
        evo_core = QUANTUM_CORE.get("evolution_trainer")
        if evo_core is None:
            payload = {"error": "Evolutionary Core not initialized or unavailable."}
            await self._send_response(
                503,
                "Service Unavailable",
                json.dumps(payload),
                content_type="application/json",
            )
            return

        steps = 100
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        if content_length > 0:
            try:
                body = await self.reader.readexactly(content_length)
                data = json.loads(body.decode("utf-8"))
                steps = int(data.get("steps", 100))
            except Exception:
                steps = 100

        steps = max(1, min(steps, 10_000))

        logger.info(
            "User %s triggered Evolutionary Race Cycle for %d steps.",
            self.session.get("user_id"),
            steps,
        )

        start = time.time()

        try:
            if hasattr(evo_core, "evolutionary_race_cycle"):
                evo_core.evolutionary_race_cycle(steps)
        except Exception as e:
            logger.error("Evolutionary core error: %s", e, exc_info=True)

        laser_core = QUANTUM_CORE.get("laser")
        if laser_core is not None:
            try:
                laser_core.log_event(0.95, f"EVOLUTION_RACE_COMPLETE steps={steps}")
                laser_core.set_coherence_level(random.uniform(0.8, 1.0))
            except Exception:
                pass

        end = time.time()

        result = {
            "message": "Evolutionary Race Cycle simulated successfully.",
            "steps_executed": steps,
            "time_ms": int((end - start) * 1000),
            "evolutionary_outcome": "Emergent Qualia Detected",
        }

        await self._send_response(
            200,
            "OK",
            json.dumps(result),
            content_type="application/json",
        )

# ---------------------------------------------------------------------------
# Quantum HTTP server wrapper
# ---------------------------------------------------------------------------


class QuantumHTTPServer:
    def __init__(self, host: str, port: int, ssl_context: Optional[ssl.SSLContext]):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            ssl=self.ssl_context,
        )

        scheme = "HTTPS" if self.ssl_context else "HTTP"
        logger.info("Quantum %s server serving on %s:%d", scheme, self.host, self.port)

        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        handler = QuantumHTTPRequestHandler(reader, writer)
        await handler.handle_request()

# ---------------------------------------------------------------------------
# Public entrypoint: run_quantum_server
# ---------------------------------------------------------------------------


async def run_quantum_server(quantum_core: Optional[Any] = None) -> None:
    """
    Entry point used by main.py:

        await run_quantum_server(quantum_core)

    The quantum_core argument is accepted for compatibility but not required.
    This HTTP daemon maintains its own QUANTUM_CORE helper instances.
    """
    global QUANTUM_CORE
    QUANTUM_CORE = {}

    if QYRINTH_MODULES_LOADED:
        try:
            QUANTUM_CORE["evolution_trainer"] = EvolutionaryEcosystem()
        except Exception as e:
            logger.error("Failed to initialize EvolutionaryEcosystem: %s", e)

        try:
            QUANTUM_CORE["qubitlearn"] = QubitLearn("httpd")
        except Exception as e:
            logger.error("Failed to initialize QubitLearn: %s", e)

        try:
            QUANTUM_CORE["laser"] = LASERUtility(parent_config={"quantum_mode": True})
        except Exception as e:
            logger.error("Failed to initialize LASERUtility: %s", e)

    ssl_ctx: Optional[ssl.SSLContext] = None
    cert_path = Path(CONTEXT_CERT)
    key_path = Path(CONTEXT_KEY)

    if cert_path.exists() and key_path.exists():
        try:
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(str(cert_path), str(key_path))
            logger.info("SSL context loaded. Running as HTTPS.")
        except Exception as e:
            logger.error("Error loading SSL context (%s). Falling back to HTTP.", e)
            ssl_ctx = None
    else:
        logger.warning(
            "SSL certificate/key not found (%s, %s). Running as plain HTTP.",
            CONTEXT_CERT,
            CONTEXT_KEY,
        )

    server = QuantumHTTPServer(SERVER_HOST, SERVER_PORT, ssl_ctx)
    logger.info("Quantum server initialized. Awaiting emergence...")
    await server.start()

# ---------------------------------------------------------------------------
# Stand-alone execution (for manual testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(run_quantum_server(None))
    except KeyboardInterrupt:
        logger.info("Quantum HTTP server interrupted by user.")
    except Exception as exc:
        logger.error("Fatal server execution error: %s", exc, exc_info=True)
