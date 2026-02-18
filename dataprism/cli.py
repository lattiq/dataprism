"""DataPrism CLI — command-line interface for EDA tools."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dataprism",
        description="DataPrism — exploratory data analysis toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    # view subcommand
    view_parser = subparsers.add_parser("view", help="Open EDA results in an interactive dashboard")
    view_parser.add_argument("file", help="Path to a JSON results file")
    view_parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765)")
    view_parser.add_argument(
        "--no-browser", action="store_true", help="Don't open the browser automatically"
    )

    args = parser.parse_args(argv)

    if args.command == "view":
        from dataprism.viewer import serve_results

        serve_results(args.file, port=args.port, open_browser=not args.no_browser)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
