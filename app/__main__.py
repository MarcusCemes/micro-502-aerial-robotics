import time
from asyncio import Event, run
from signal import CTRL_C_EVENT, SIGINT, signal
from sys import exit, stderr

from loguru import logger

from cflib.crtp import init_drivers

from .bigger_brain import BiggerBrain
from .common import Context
from .drone import Drone

EXIT_SIGNALS = [SIGINT, CTRL_C_EVENT]


@logger.catch
def main():
    logger.remove()
    logger.add(
        stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    try:
        run(init())

    except KeyboardInterrupt:
        exit(0)

    finally:
        quit()


async def init():
    init_drivers()

    logger.info("Connecting to drone...")
    data_event = Event()
    drone = Drone(data_event)

    await drone.connect()
    await drone.reset_estimator()

    drone.configure_logging()

    ctx = Context(drone, data_event)

    try:
        await BiggerBrain(ctx).run()

    finally:
        ctx.drone.disconnect()


def install_signal_handlers():
    for sig in EXIT_SIGNALS:
        signal(sig, quit)


def quit(*_) -> None:
    time.sleep(3)
    logger.warning("Process did not exit after 3 seconds, forcing exit...")
    exit(1)


if __name__ == "__main__":
    main()
