"""``python -m monorl.server [--host H] [--port P] [--open]``."""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="monorl.server", description="Run the monorl web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open", action="store_true", help="Open the UI in a browser once the server starts."
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    import uvicorn

    from monorl.server.app import create_app

    app = create_app()
    if args.open:
        threading.Timer(1.0, webbrowser.open, args=(f"http://{args.host}:{args.port}/",)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
