"""Run the GUI with no arguments, or the CLI when arguments are supplied."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
