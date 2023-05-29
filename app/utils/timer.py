from time import time


class Timer:
    def __init__(self):
        self._time = 0.0

    def reset(self) -> None:
        self._time = time()

    def is_elapsed(self, duration: float) -> bool:
        """
        Returns:
            True if the timer has waited duration
        """
        return time() - self._time >= duration
