import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import Processor

from src.config.settings import LogFormat, LoggingSettings



def configure_logging(settings: LoggingSettings) -> None:
    settings.directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(
            fmt="iso",
            utc=True,
        ),
        structlog.processors.format_exc_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.level.value)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.level.value,
            "formatter": "structured",
            "stream": "ext://sys.stderr",
        },
    }

    root_handlers = ["console"]

    if settings.log_to_file:
        log_file = settings.directory / settings.filename

        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.level.value,
            "formatter": "structured",
            "filename": str(log_file),
            "maxBytes": settings.max_bytes,
            "backupCount": settings.backup_count,
            "encoding": "utf-8",
        }

        root_handlers.append("file")

    console_renderer = structlog.dev.ConsoleRenderer()
    json_renderer = structlog.processors.JSONRenderer()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": console_renderer,
                    "foreign_pre_chain": shared_processors,
                },
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": json_renderer,
                    "foreign_pre_chain": shared_processors,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": settings.level.value,
                    "formatter": "console",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": settings.level.value,
                    "formatter": "json",
                    "filename": str(
                        settings.directory / settings.filename
                    ),
                    "maxBytes": settings.max_bytes,
                    "backupCount": settings.backup_count,
                    "encoding": "utf-8",
                },
            },
            "root": {
                "handlers": ["console", "file"],
                "level": settings.level.value,
            },
        }
    )
