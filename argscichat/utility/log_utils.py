import logging
import os
import sys
from logging import FileHandler
import traceback

import argscichat.const_define as cd

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


class SimpleLevelFilter(object):
    """
    Simple logging filter
    """

    def __init__(self, level):
        self._level = level

    def filter(self, log_record):
        """
        Filters log message according to filter level

        :param log_record: message to log
        :return: True if message level is less than or equal to filter level
        """

        return log_record.levelno <= self._level


class Logger(object):
    _instance = None
    _log_path = None

    @classmethod
    def _handle_exception(cls, exctype, value, tb):
        if cls._instance is not None:
            cls._instance.info("Type: {0}\n"
                               "Value: {1}\n"
                               "Traceback: {2}\n".format(exctype, value, ''.join(traceback.format_exception(exctype, value, tb))))

    @classmethod
    def _build_logger(cls, name):
        """
        Returns a logger instance that handles info, debug, warning and error messages.

        :param name: logger name
        :return: logger instance
        """

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

        if cls._log_path is None:
            log_path = os.path.join(cd.PATH_LOG, cd.NAME_LOG)

            if not os.path.isdir(cd.PATH_LOG):
                os.makedirs(cd.PATH_LOG)

        else:
            log_path = os.path.join(cls._log_path, cd.NAME_LOG)

            if not os.path.isdir(cls._log_path):
                os.makedirs(cls._log_path)

        trf_handler = FileHandler(log_path)
        trf_handler.setLevel(logging.DEBUG)
        trf_handler.setFormatter(formatter)
        logger.addHandler(trf_handler)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(SimpleLevelFilter(logging.WARNING))
        logger.addHandler(stdout_handler)

        sys.excepthook = cls._handle_exception

        return logger

    @classmethod
    def set_log_path(cls, log_path=None):
        cls._log_path = log_path

    @classmethod
    def get_logger(cls, name):
        if cls._instance is None:
            print("[{0}] Retrieving new logger: {1}".format(cls.__name__, cls._log_path))
            cls._instance = cls._build_logger(name)
        return cls._instance
