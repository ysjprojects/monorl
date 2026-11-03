"""FastAPI web server: job manager (subprocess ``python -m monorl.cli train``), read-only run
directory API, Hub/dataset browsing and the static web UI. Torch-free at import."""

from monorl.server.app import create_app

__all__ = ["create_app"]
