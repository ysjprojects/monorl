from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("openenv.core")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@pytest.fixture(scope="module")
def reverse_server_url():
    """A real openenv-core http/websocket server for ReverseEnv, in a daemon thread."""
    import uvicorn
    from openenv.core.env_server.http_server import create_fastapi_app
    from sample_envs import ReverseAction, ReverseEnv, ReverseObservation

    app = create_fastapi_app(ReverseEnv, ReverseAction, ReverseObservation, max_concurrent_envs=4)
    # openenv's /ws handler races the client's close and logs a WebSocketDisconnect traceback.
    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise TimeoutError("openenv server did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
