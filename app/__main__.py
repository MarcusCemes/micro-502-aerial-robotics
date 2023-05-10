import time
from asyncio import run
from signal import CTRL_C_EVENT, SIGINT, signal
from sys import exit, stderr

from loguru import logger

from cflib.crtp import init_drivers

from .big_brain import BigBrain
from .context import Context

EXIT_SIGNALS = [SIGINT, CTRL_C_EVENT]


@logger.catch
def main():
    logger.remove()
    logger.add(
        stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    run(init())
    quit()


async def init():
    init_drivers()

    ctx = Context()

    logger.info("Connecting to drone...")
    await ctx.drone.connect()
    await ctx.drone.reset_estimator()

    ctx.drone.configure_logging()

    try:
        await BigBrain(ctx).run()

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
