import threading
from asyncio import Event, run
from os import _exit
from signal import CTRL_C_EVENT, SIGINT, signal
from sys import _current_frames, stderr
from time import sleep
from traceback import print_stack
from typing import Final

from loguru import logger

from cflib.crtp import init_drivers

from .bigger_brain import BiggerBrain
from .common import Context
from .drone import Drone

EXIT_SIGNALS: Final = [SIGINT, CTRL_C_EVENT]
TIMEOUT_S: Final = 3


def main():
    logger.remove()
    logger.add(
        stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    try:
        run(init())

    except KeyboardInterrupt:
        logger.info("✋ Interrupt handled")

    except ConnectionError as e:
        logger.error(f"Failed to connect to drone")

    except Exception as e:
        logger.exception(f"Application error: {e}")

    finally:
        StopWatchdog().start()


async def init():
    init_drivers()

    data_event = Event()

    async with Drone(data_event) as drone:
        await drone.reset_estimator()
        drone.configure_logging()

        ctx = Context(drone, data_event)

        await BiggerBrain(ctx).run()


class StopWatchdog(threading.Thread):
    def __init__(self, timeout=3):
        super().__init__()

        self.daemon = True
        self._timeout = timeout

    def run(self):
        sleep(self._timeout)

        logger.warning("Process did not exit, the following threads are alive:")

        frames = _current_frames()
        for thread in threading.enumerate():
            if thread.is_alive() and not thread.daemon:
                logger.warning(f"  {thread.name}")
                if thread.ident is not None:
                    print_stack(frames[thread.ident])

        _exit(1)


if __name__ == "__main__":
    main()
