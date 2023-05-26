from asyncio import Queue, QueueEmpty, create_task, wait
from dataclasses import asdict
from enum import Enum

from loguru import logger

from cflib.positioning.motion_commander import MotionCommander

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
        mctl = MotionCommander(self._ctx.drone.cf)

        try:
            logger.info("🚁 Taking off")
            mctl.take_off()

            debug_counter = 0

            while True:
                debug_counter += 1

                if debug_counter == 4:
                    debug_counter = 0
                    self._ctx.debug_tick = True

                if self._ctx.new_data.is_set():
                    logger.warning("🐢 Too slow, missed sensor data events!")

                maybe_cmd = await self._wait_for_event(cmds)

                self._ctx.outlet.broadcast(
                    {"type": "sensors", "data": asdict(self._ctx.sensors)}
                )
                if self._ctx.drone.first_landing:
                    logger.debug("🚁 First landing")
                    await self._ctx.drone.comeback(self._ctx.drone.new_pos.x, self._ctx.drone.new_pos.y)
                    self._ctx.drone.first_landing = False

                # s = self._ctx.sensors
                # print(
                #     f"f {s.front:.2f} l {s.left:.2f} b {s.back:.2f} r {s.right:.2f} y {s.yaw:.2f}"
                # )

                if maybe_cmd is not None:
                    self._handle_cmd(maybe_cmd, mctl)
                    return

                self._ctx.new_data.clear()

                self._update_sensors()
                self._nav.update()

                if self._flight_ctl.update():
                    break

                self._flight_ctl.apply_flight_command(mctl)

                self._ctx.debug_tick = False

            logger.info("🚁 Landing")
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

    def _handle_cmd(self, cmd: Command, mctl: MotionCommander) -> None:
        match cmd:
            case Command.Land:
                logger.info("🤚 Emergency landing")
                mctl.land()
                logger.info("👍 Landed")

            case Command.Stop:
                logger.info("⛔ Emergency stop")
