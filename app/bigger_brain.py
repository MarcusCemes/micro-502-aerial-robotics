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
    """
    The BiggerBrain class is responsible for the high-level control of the application.
    An infinite loop is repeatedly evaluated that reads drone sensor data, updates concerned
    modules (such as Navigation), evaluates the Finite State Machine (FlightController)
    and generates the latest flight command to transmit to the drone.
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

        self._nav = Navigation(self._ctx)
        self._flight_ctl = FlightController(ctx, self._nav)

    async def run(self, cmds: Queue[Command]) -> None:
        try:
            # The following counter is used to reduce the frequency of some operations
            # to run every few ticks. When the debug counter overflows, a flag is set
            # in the shared Context class.
            debug_counter = 0

            while True:
                debug_counter += 1

                if debug_counter == 4:
                    debug_counter = 0
                    self._ctx.debug_tick = True

                # Release the event loop until sensor data is available or
                # a terminal command is received.
                maybe_cmd = await self._wait_for_event(cmds)
                self._ctx.new_data.clear()

                if maybe_cmd is not None:
                    await self._handle_cmd(maybe_cmd)
                    return

                # Fetch the latest sensor data from the receiver thread and update
                # the Sensors dataclass in the Context class.
                self._update_sensors()

                # Transmit the sensor data to a running HTTP WebSocket server
                self._ctx.outlet.broadcast(
                    {"type": "sensors", "data": asdict(self._ctx.sensors)}
                )

                # Update the obstacle map in Navigation (reads Sensors from Context)
                self._nav.update()

                # Evaluate the FlightController finite state machine
                if self._flight_ctl.update():
                    break

                # Generate the flight command and transmit to drone
                self._flight_ctl.apply_flight_command()

                self._ctx.debug_tick = False

        finally:
            self._ctx.drone.cf.commander.send_stop_setpoint()

    # == Private == #

    def _update_sensors(self) -> None:
        """
        Retrieves the latset sensor data from the Drone class using a mutex.
        """

        reading = self._ctx.drone.get_last_sensor_reading()
        self._ctx.sensors = reading

    async def _wait_for_event(self, cmds: Queue[Command]) -> Command | None:
        """
        Runs two coroutines concurrently waiting for different events, returning
        as soon as one of them is resolved (equivilent to Promise.race()).
        """

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

            case Command.Stop:
                logger.info("⛔ Emergency stop")
