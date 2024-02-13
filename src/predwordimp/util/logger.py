import logging


def get_logger(
    module_name: str | None = None, log_level: int = logging.INFO
) -> logging.Logger:
    if module_name is None:
        module_name = __name__

    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
