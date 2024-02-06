import logging
import sys


def setup_logger() -> None:
    log_format = '[%(levelname)s] %(asctime)s %(filename)s (%(lineno)d)\t- %(message)s'
    log_dateformat = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=log_format, datefmt=log_dateformat, stream=sys.stdout, level=logging.INFO)
