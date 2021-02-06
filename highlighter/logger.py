import logging

LOG_LEVEL = logging.WARNING


class TensorflowFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("tensorflow")


def configure_logger():
    log = logging.getLogger()
    log.propagate = False
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s (line %(lineno)d): %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(TensorflowFilter())
    stream_handler.setLevel(LOG_LEVEL)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    log.setLevel(LOG_LEVEL)
