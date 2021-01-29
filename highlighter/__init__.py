import os

from . import logger
from .version import __version__

#  Level | Level for Humans | Level Description
#  -------|------------------|------------------------------------
#   0     | DEBUG            | [Default] Print all messages
#   1     | INFO             | Filter out INFO messages
#   2     | WARNING          | Filter out INFO & WARNING messages
#   3     | ERROR            | Filter out all messages


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger.configure_logger()
