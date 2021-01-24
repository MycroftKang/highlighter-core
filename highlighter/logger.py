import logging

LOG_LEVEL = logging.WARNING


def configure_logger():
    log = logging.getLogger()
    log.propagate = False
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s (line %(lineno)d): %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    log.setLevel(LOG_LEVEL)
