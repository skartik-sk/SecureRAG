import logging
import os


def setup_logging(default_level: str = "WARNING") -> None:
    """Route app logs to stderr at LOG_LEVEL.

    LOG_LEVEL=INFO turns on the RAG trace lines (one structured line per graph
    node: node=grade graded=1 kept=2 attempt=0 elapsed_ms=412). Safe to call
    repeatedly: configures the root logger once, later calls just apply the
    level.
    """
    level = os.getenv("LOG_LEVEL", default_level).upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)  # already configured (uvicorn etc.) — just apply the level
    else:
        logging.basicConfig(level=level,
                            format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)