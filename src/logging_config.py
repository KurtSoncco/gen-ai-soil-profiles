import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Set up a basic logging configuration.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
