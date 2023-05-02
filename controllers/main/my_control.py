import matplotlib.image
import numpy as np

from common import Context, Sensors
from config import DEBUG_MAP, SERVER_ENABLED
from flight_ctl import FlightController
from log import Logger
from navigation import Navigation


class MyController(Logger):
    def __init__(self):
        ctx = Context()

        self.ctx = ctx
        self.nav = Navigation(ctx)
        self.flight_ctl = FlightController(ctx)
        self.server = None

        if SERVER_ENABLED:
            self._enable_server()

    def step_control(self, data: dict[str, float]) -> list[float]:
        try:
            ctx = self.ctx

            ctx.sensors = Sensors(**data)
            ctx.outlet.broadcast({"type": "sensors", "data": data})

            cmd = self.flight_ctl.update()

            if DEBUG_MAP and self.ctx.debug_tick:
                img = np.flip(np.flip(self.nav.map, 1), 0)
                matplotlib.image.imsave("map.png", img, cmap="gray")

            return cmd.to_list()
        except Exception as e:
            self.error(e)
            raise e

    def _enable_server(self):
        from server import Server

        server = Server(self.ctx)
        server.start()

        self.server = server

    def destroy(self):
        if self.server is not None:
            self.server.stop()
