import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "create_run_dir",
    "configure_logger",
    "get_logger",
    "log_metrics",
]

_LOGGER: Optional[logging.Logger] = None


def create_run_dir(
    experiment: str,
    base_dir: os.PathLike[str] | str = "runs",
    config_path: os.PathLike[str] | str | None = None,
    metadata: Dict[str, Any] | None = None,
    copy_config: bool = True,
    write_metadata: bool = True,
    timestamp_fmt: str = "%Y%m%d-%H%M%S",
) -> Path:
    """
    Create a timestamped run directory under a base path.

    Args:
        experiment: Name for the experiment; used in the folder name.
        base_dir: Root directory under which runs are created.
        config_path: Optional path to a YAML config to copy into the run folder.
        metadata: Optional dictionary to serialize as metadata.json.
        copy_config: When True, copy ``config_path`` into the run folder if provided.
        write_metadata: When True, write metadata.json using ``metadata``.
        timestamp_fmt: strftime format for the timestamp appended to the folder.

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime(timestamp_fmt)
    run_dir = Path(base_dir).expanduser().resolve() / f"{experiment}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if copy_config and config_path:
        source = Path(config_path)
        if source.exists():
            destination = run_dir / source.name
            destination.write_bytes(source.read_bytes())

    if write_metadata:
        meta = dict(metadata or {})
        meta.setdefault("experiment", experiment)
        meta.setdefault("timestamp", timestamp)
        meta.setdefault("run_dir", str(run_dir))
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return run_dir


def configure_logger(
    run_dir: os.PathLike[str] | str,
    name: str = "pinn",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger that writes to stdout and a file in the run directory.

    Args:
        run_dir: Directory where the log file will be written.
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    log_file = run_path / "run.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    _LOGGER = logger
    return logger


def get_logger(name: str = "pinn") -> logging.Logger:
    """
    Retrieve the configured logger, or fall back to a standard logger.

    Args:
        name: Logger name to retrieve when not yet configured.
    """
    return _LOGGER if _LOGGER is not None else logging.getLogger(name)


def log_metrics(
    metrics: Dict[str, Any],
    run_dir: os.PathLike[str] | str,
    filename: str = "metrics.csv",
    fmt: str = "csv",
) -> None:
    """
    Append scalar metrics to a CSV or JSONL file.

    Args:
        metrics: Dictionary of metric name to value.
        run_dir: Run directory where the metrics file is stored.
        filename: Metrics file name inside the run directory.
        fmt: File format, either ``"csv"`` or ``"jsonl"``.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / filename

    if fmt.lower() == "jsonl":
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics))
            f.write("\n")
        return

    # Default to CSV
    write_header = not metrics_path.exists()
    fieldnames = list(metrics.keys())
    with metrics_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
