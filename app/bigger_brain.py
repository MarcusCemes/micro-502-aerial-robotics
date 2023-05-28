from asyncio import Queue, QueueEmpty, create_task, sleep, wait
from dataclasses import asdict
from enum import Enum
from time import time

from loguru import logger

from .common import Context
from .flight_ctl import FlightController
from .navigation import Navigation


class Command(Enum):
    Land = "land"
    Stop = "stop"


class BiggerBrain:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

        self._nav = Navigation(self._ctx)

        self._flight_ctl = FlightController(ctx, self._nav)
        self._last_execution = 0.0

    async def run(self, cmds: Queue[Command]) -> None:
        try:
            logger.info("🚁 Taking off")

            debug_counter = 0

            while True:
                debug_counter += 1

                if debug_counter == 4:
                    debug_counter = 0
                    self._ctx.debug_tick = True

                # Free the event loop to allow sensors to be updated
                await sleep(1e-3)

                # if self._ctx.new_data.is_set():
                #     logger.warning("🐢 Too slow, missed sensor data events!")

                maybe_cmd = await self._wait_for_event(cmds)
                self._ctx.new_data.clear()

                # start_time = time()
                self._update_sensors()

                self._ctx.outlet.broadcast(
                    {"type": "sensors", "data": asdict(self._ctx.sensors)}
                )

                # if self._ctx.drone.first_landing:
                #     logger.debug("🚁 First landing")
                #     await self._ctx.drone.reset_estimator(self._ctx.drone.new_pos)
                #     self._ctx.drone.first_landing = False

                if maybe_cmd is not None:
                    await self._handle_cmd(maybe_cmd)
                    return

                self._nav.update()

                if self._flight_ctl.update():
                    break

                self._flight_ctl.apply_flight_command()

                # end_time = time()
                # duration = end_time - start_time

                # if duration >= 20e-3:
                #     logger.warning(f"Brain took {1e3 * duration:.2f} ms")

                self._ctx.debug_tick = False

            logger.info("🚁 Landing")

        finally:
            self._ctx.drone.cf.commander.send_stop_setpoint()
            self._nav.stop()

    # == Private == #

    def _update_sensors(self) -> None:
        reading = self._ctx.drone.get_last_sensor_reading()
        self._ctx.sensors = reading

    async def _wait_for_event(self, cmds: Queue[Command]) -> Command | None:
        try:
            return cmds.get_nowait()

        except QueueEmpty:
            pass

        new_data_co = self._ctx.new_data.wait()
        cmd_co = cmds.get()

        tasks = [create_task(co) for co in (new_data_co, cmd_co)]
        (done, pending) = await wait(tasks, return_when="FIRST_COMPLETED")

        for task in pending:
            task.cancel()

        for task in done:
            result = task.result()

            if isinstance(result, Command):
                return result

        return None

    async def _handle_cmd(self, cmd: Command) -> None:
        match cmd:
            case Command.Land:
                logger.info("🤚 Emergency landing")
                self._ctx.drone.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.0)
                await sleep(0.5)
                logger.info("👍 Landed")

            case Command.Stop:
                logger.info("⛔ Emergency stop")
