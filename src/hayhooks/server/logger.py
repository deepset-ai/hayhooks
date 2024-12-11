# logger.py

import os
import sys
from loguru import logger as log

log.remove()
log.add(sys.stderr, level=os.getenv("LOG", "INFO").upper())
