from traceback import format_exc

from sim.common import Context, Sensors
from sim.config import EXPENSIVE_COMPUTATION_TICKS, SERVER_ENABLED
from sim.flight_ctl import FlightController
from sim.log import Logger


class MyController(Logger):
    def __init__(self):
        ctx = Context()

        self.ctx = ctx
        self.flight_ctl = FlightController(ctx)
        self.server = None

        self.dead = False
        self.debug_counter = 0

        if SERVER_ENABLED:
            self._enable_server()

    def step_control(self, data: dict[str, float]) -> list[float]:
        if self.dead:
            return [0.0, 0.0, 0.0, 0.0]

        try:
            ctx = self.ctx

            ctx.sensors = Sensors(**data)
            ctx.outlet.broadcast({"type": "sensors", "data": data})

            ctx.ticks += 1
            self.debug_counter += 1

            if self.debug_counter == EXPENSIVE_COMPUTATION_TICKS:
                self.debug_counter = 0
                ctx.debug_tick = True

            cmd = self.flight_ctl.update()

            ctx.debug_tick = False

            return cmd.to_list()
        except Exception:
            self.error("Uncaught exception!\n{}".format(format_exc()))
            self.dead = True
            return [0.0, 0.0, 0.0, 0.0]

    def _enable_server(self):
        from sim.server import Server

        server = Server(self.ctx)
        server.start()

        self.server = server

    def destroy(self):
        if self.server is not None:
            self.server.stop()
