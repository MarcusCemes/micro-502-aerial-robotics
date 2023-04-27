# == ASCII Escape Codes == #

RESET = "\u001b[0m"
BOLD = "\u001b[0m"
RED = "\u001b[31m"
GREEN = "\u001b[32m"
YELLOW = "\u001b[33m"
BLUE = "\u001b[34m"


class Logger:
    def info(self, msg: str):
        self.__log(msg, "INFO ", GREEN)

    def warn(self, msg: str):
        self.__log(msg, "WARN ", YELLOW)

    def error(self, msg: str):
        self.__log(msg, "ERROR", RED)

    def __log(self, msg: str, level: str, color: str):
        print(f"{color}{level}{RESET} [{self.__class__.__name__}] {msg}")
