class Getch:
    """
    Reads raw terminal input using platform-specific implementations.
    """
    
    def __init__(self):
        try:
            self._impl = _GetchWindows()

        except ImportError:
            self._impl = _GetchUnix()

    def __call__(self):
        return self._impl()


class _GetchUnix:
    def __init__(self):
        import sys
        import tty

        self._tty = tty
        self._sys = sys

    def __call__(self):
        import termios

        fd = self._sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)  # type: ignore

        try:
            self._tty.setraw(self._sys.stdin.fileno())  # type: ignore
            return self._sys.stdin.read(1)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # type: ignore


class _GetchWindows:
    def __init__(self):
        import msvcrt

        self._msvcrt = msvcrt

    def __call__(self):
        return self._msvcrt.getch()
