"""Logging set-up for the command-line entry points (library modules only call getLogger)."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Print log records to the console without prefixes, like the former progress messages.

    Called once from each `main()`. Library code never configures logging itself, so importing a
    module from a notebook, a test or the dashboard changes nothing.
    """
    logging.basicConfig(level=level, format="%(message)s")
