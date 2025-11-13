import logging
import time, os
from pathlib import Path

# Project root directory
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
LOG_FILE_PATH = PROJECT_ROOT_PATH / "src" / "data" / "logs"

class MineLogger:
    def __init__(self, log_path=LOG_FILE_PATH, file_level=logging.DEBUG, console_level=logging.INFO):
        # Timestamp format compatible with Windows file names
        time_format = "%Y-%m-%d-%H-%M-%S"
        time_str = time.strftime(time_format, time.localtime())

        # Logger configuration
        # Log message format
        LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

        # Logging levels
        CONSOLE_LOG_LEVEL = console_level
        FILE_LOG_LEVEL = file_level

        # Console handler
        self.CONSOLE_HANDLER = logging.StreamHandler()
        self.CONSOLE_HANDLER.setLevel(CONSOLE_LOG_LEVEL)
        self.CONSOLE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))

        # File handler
        self.log_path = Path(log_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.FILE_HANDLER = logging.FileHandler(self.log_path / f'openmines_sim_{time_str}.log')
        self.FILE_HANDLER.setLevel(FILE_LOG_LEVEL)
        self.FILE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))

    def get_logger(self, name):
        """Return a module-level logger after handlers are in place."""
        logger = logging.getLogger(name)
        # Attach console handler only once
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            logger.addHandler(self.CONSOLE_HANDLER)
        # Attach file handler only once
        if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
            logger.addHandler(self.FILE_HANDLER)
        logger.setLevel(logging.DEBUG)
        return logger