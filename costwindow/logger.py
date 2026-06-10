import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_local_logger() -> logging.Logger:
    """Returns a logger that writes warnings to ~/.costwindow/logs/ with daily rotation."""
    log_dir = Path.home() / ".costwindow" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("costwindow.fallback")
    if not logger.handlers:
        handler = TimedRotatingFileHandler(
            log_dir / "costwindow.log",
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger
