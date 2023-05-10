from asyncio import sleep
from time import time
from typing import Final

from .config import TARGET_FREQUENCY
from .context import Context
from .flight_ctl import FlightController

CONTROL_PERIOD: Final = 1 / TARGET_FREQUENCY


class BigBrain:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

        self._flight_ctl = FlightController(ctx)
        self._last_execution = 0.0

    async def run(self) -> None:
        while True:
            self._last_execution = time()

            self._update_sensors()

            self._flight_ctl.update()
            self._flight_ctl.apply_flight_command()

            await self._sleep_until_next_execution()

    # == Private == #

    def _update_sensors(self) -> None:
        reading = self._ctx.drone.get_last_sensor_reading()

        if reading is not None:
            self._ctx.sensors = reading

    async def _sleep_until_next_execution(self) -> None:
        duration = min(CONTROL_PERIOD, time() - self._last_execution)
        await sleep(duration)
