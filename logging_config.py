import logging
import sys


class AppLogger(logging.Logger):
    def _log(self, level, msg, args, **kwargs):
        if kwargs.pop("highlight", False):
            msg = f"\n\n{'=' * 50}\n{msg}\n{'=' * 50}"
        super()._log(level, msg, args, **kwargs)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        # logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        message = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        reset = self.RESET if color else ""
        return f"{color}{message}{reset}"


logging.setLoggerClass(AppLogger)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(message)s"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(handler)


def getLogger(name):
    return logging.getLogger(name)

def setLevel(level):
    level = getattr(logging, level.upper())
    root_logger.setLevel(level)
    handler.setLevel(level)
