import threading
from asyncio import (
    Event,
    Queue,
    create_task,
    get_event_loop,
    run,
    run_coroutine_threadsafe,
)
from os import _exit
from signal import CTRL_C_EVENT, SIGINT
from sys import _current_frames, stderr
from time import sleep
from traceback import print_stack
from typing import Final

from loguru import logger
from app.server import Server

from app.utils.getch import Getch
from cflib.crtp import init_drivers

from .bigger_brain import BiggerBrain, Command
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

    cmds = Queue[Command]()
    CommandReader(cmds).start()

    async with Drone(data_event) as drone:
        await drone.reset_estimator()
        drone.configure_logging()

        ctx = Context(drone, data_event)

        # server = Server(ctx)
        # stop_server = Event()
        # server_task = create_task(server.run(stop_server))

        await BiggerBrain(ctx).run(cmds)

        # stop_server.set()
        # await server_task


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


class CommandReader(threading.Thread):
    def __init__(self, cmds: Queue[Command]):
        super().__init__()

        self.daemon = True
        self._queue = cmds
        self._loop = get_event_loop()
        self._stop = False

    def run(self) -> None:
        getch = Getch()

        while not self._stop:
            match getch():
                case b"\x03" | b"\x04" | b"s":
                    self._send_cmd(Command.Stop)
                    return

                case b"l":
                    self._send_cmd(Command.Land)
                    return

                case _:
                    pass

    def stop(self):
        self._stop = True

    def _send_cmd(self, cmd: Command) -> None:
        run_coroutine_threadsafe(self._queue.put(cmd), self._loop)


def read_char():
    try:
        # for Windows-based systems
        import msvcrt

        return msvcrt.getch()

    except ImportError:
        # for POSIX-based systems (with termios & tty support)
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        oldSettings = termios.tcgetattr(fd)  # type: ignore

        try:
            tty.setcbreak(fd)  # type: ignore
            answer = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, oldSettings)  # type: ignore

        return answer


if __name__ == "__main__":
    main()
