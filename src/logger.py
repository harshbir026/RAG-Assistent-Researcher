import logging
import json
import time
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── structured query logger ──────────────────────────────
# Separate from Python's standard logging — this writes one
# JSON line per query to logs/queries.jsonl, designed to be
# grep-able and easy to analyze later (e.g. for Day 14's
# latency measurement).
QUERY_LOG_PATH = LOG_DIR / "queries.jsonl"


def log_query(
    query: str,
    chunk_ids: list[str],
    answer_length: int,
    latency_seconds: float,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """
    Log one query as a JSON line. Called after every retrieve+generate
    cycle, whether it succeeded or failed.
    """
    entry = {
        "timestamp": time.time(),
        "query": query,
        "chunk_ids": chunk_ids,
        "answer_length": answer_length,
        "latency_seconds": round(latency_seconds, 3),
        "status": status,
        "error": error,
    }
    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── standard application logger ──────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Standard Python logger for general application events
    (errors, warnings, retries). Logs to both console and
    logs/app.log.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on reimport
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_DIR / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger