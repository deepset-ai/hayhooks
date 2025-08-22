import os
import sys

from loguru import logger as log


def formatter(record):
    if record["extra"]:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level> - <magenta>{extra}</magenta>\n"
        )

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>"
        "{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>\n"
    )


log.remove()
log.add(sys.stderr, level=os.getenv("LOG", "INFO").upper(), format=formatter)
