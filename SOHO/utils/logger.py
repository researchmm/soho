import logging

from commons.runner import get_dist_info

def get_root_logger(log_file=None,log_level=logging.INFO):
    """Get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "SOHO".
    
    :param log_file: 
    :param log_level: 
    :return: 
    """
    logger = logging.getLogger(__name__.split('.')[0])

    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger

def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used. Some
            special loggers are:
            - "root": the root logger obtained with `get_root_logger()`.
            - "silent": no message will be printed.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif logger == 'root':
        _logger = get_root_logger()
        _logger.log(level, msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger != 'silent':
        raise TypeError(
            'logger should be either a logging.Logger object, "root", '
            '"silent" or None, but got {}'.format(logger))

