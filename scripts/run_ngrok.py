from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pyngrok import ngrok
from pyngrok.conf import PyngrokConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got: {raw_value!r}") from exc


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    authtoken = os.getenv("NGROK_AUTHTOKEN")
    if not authtoken:
        raise RuntimeError("Set NGROK_AUTHTOKEN in .env before running this script.")

    port = _get_env_int("NGROK_PORT", 8000)
    domain = os.getenv("NGROK_DOMAIN") or None
    ngrok_path = Path(sys.executable).resolve().parent / "ngrok.exe"

    pyngrok_config = PyngrokConfig(ngrok_path=str(ngrok_path))
    ngrok.set_auth_token(authtoken, pyngrok_config=pyngrok_config)

    server = None
    started_server = False
    if _is_port_open("127.0.0.1", port):
        print(f"Port {port} is already in use. Reusing the existing local server.")
    else:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            cwd=PROJECT_ROOT,
        )
        started_server = True

    tunnel = None
    try:
        if server is not None:
            time.sleep(2)
            if server.poll() is not None:
                return server.returncode or 1

        connect_kwargs = {"addr": port, "proto": "http"}
        if domain:
            connect_kwargs["domain"] = domain
        tunnel = ngrok.connect(**connect_kwargs, pyngrok_config=pyngrok_config)

        print()
        print("FastAPI server is running.")
        print(f"Local docs:  http://127.0.0.1:{port}/docs")
        print(f"ngrok URL:   {tunnel.public_url}")
        print(f"ngrok docs:  {tunnel.public_url}/docs")
        print()
        print("Press Ctrl+C to stop the server and tunnel.")

        if server is not None:
            server.wait()
            return server.returncode or 0

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopping server and ngrok tunnel...")
        return 0
    finally:
        if tunnel is not None:
            ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        if started_server and server is not None and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
