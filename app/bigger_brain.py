from loguru import logger

from cflib.positioning.motion_commander import MotionCommander

from .common import Context
from .flight_ctl import FlightController


class BiggerBrain:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

        self._flight_ctl = FlightController(ctx)
        self._last_execution = 0.0

    async def run(self) -> None:
        mctl = MotionCommander(self._ctx.drone.cf)

        try:
            mctl.take_off()

            debug_counter = 0

            while True:
                debug_counter += 1

                if debug_counter == 4:
                    debug_counter = 0
                    self._ctx.debug_tick = True

                await self._ctx.new_data.wait()

                self._ctx.new_data.clear()

                self._update_sensors()

                if self._flight_ctl.update():
                    break

                self._flight_ctl.apply_flight_command(mctl)

                self._ctx.debug_tick = False

            mctl.land()

        finally:
            if mctl._thread is not None:
                mctl._thread.stop()

            self._ctx.drone.cf.commander.send_stop_setpoint()

    # == Private == #

    def _update_sensors(self) -> None:
        reading = self._ctx.drone.get_last_sensor_reading()

        if reading is not None:
            self._ctx.sensors = reading
