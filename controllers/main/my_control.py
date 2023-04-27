from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import matplotlib.image

from config import SERVER_ENABLED
from log import Logger
from common import Context, Sensors
from navigation import Navigation


AIRBORN_THRESHOLD = 0.48
TARGET_HEIGHT = 0.5


@dataclass
class ControlCommand:
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    yaw_rate: float = 0.0
    altitude: float = 0.0

    def to_list(self):
        return [self.velocity_x, self.velocity_y, self.yaw_rate, self.altitude]


class State(Enum):
    Boot = 0
    TakeOff = 1
    Hover = 2
    Land = 3


@dataclass
class Transition:
    state: State
    rerun: bool = True


Result = Optional[Transition]


class MyController(Logger):
    def __init__(self):
        ctx = Context()

        self.ctx = ctx
        self.nav = Navigation(ctx)
        self.server = None

        if SERVER_ENABLED:
            from server import Server

            server = Server(ctx)
            server.start()

            self.server = server

    def step_control(self, data: dict[str, float]):
        self.ctx.sensors = Sensors(**data)

        self.nav.update()

        # if self.ctx.debug_tick:
        #     img = np.flip(np.flip(self.nav.map, 1), 0)
        #     matplotlib.image.imsave("map.png", img, cmap="gray")

        return [0.0, 0.0, 0.0, 0.2]

    def destroy(self):
        self.info("Cleaning up...")

        if self.server is not None:
            self.server.stop()


#         while True:
#             result = self.next(data)

#             if result is not None:
#                 self.update_state(result.state)

#             if result is not None and result.rerun:
#                 continue

#             return self.flight_ctl.serialize()

#     def next(self, sensors: Sensors) -> Result:
#         match self.state:
#             case State.Boot:
#                 return self.boot()

#             case State.TakeOff:
#                 return self.takeoff(sensors)

#             case State.Hover:
#                 return self.hover(sensors)

#             case State.Land:
#                 return self.land()

#     # == State functions == #

#     def boot(self) -> Result:
#         self.info("Booting up...")
#         return Transition(State.TakeOff, rerun=True)

#     def takeoff(self, sensors: Sensors) -> Result:
#         if self.is_airborn(sensors):
#             return Transition(State.Hover)

#         self.flight_ctl.set_altitude(TARGET_HEIGHT)

#     def hover(self, sensors: Sensors) -> Result:
#         v_x, v_y = (-0.2, 0.0) if self.obstacle_present(sensors) else (0.0, 0.5)

#         self.flight_ctl.set_lateral_velocity(v_x)
#         self.flight_ctl.set_forward_velocity(v_y)

#     def land(self) -> Result:
#         self.flight_ctl.set_altitude(0.0)

#     # == Utility == #

#     def is_airborn(self, sensors: Sensors) -> bool:
#         return sensors["range_down"] >= AIRBORN_THRESHOLD

#     def obstacle_present(self, sensors: Sensors) -> bool:
#         return sensors["range_front"] < 0.2

#     def update_state(self, state: State):
#         print(f"[STATE] {state}")
#         self.state = state


# class FlightControl:
#     def __init__(self):
#         self.cmd = ControlCommand()

#     def serialize(self):
#         return self.cmd.to_list()

#     def set_forward_velocity(self, velocity: float):
#         if velocity != self.cmd.velocity_x:
#             print(f"[FLIGHT_CTL] Forward velocity: {velocity}")

#         self.cmd.velocity_x = velocity

#     def set_lateral_velocity(self, velocity: float):
#         if velocity != self.cmd.velocity_y:
#             print(f"[FLIGHT_CTL] Lateral velocity: {velocity}")

#         self.cmd.velocity_y = velocity

#     def set_altitude(self, altitude: float):
#         if altitude != self.cmd.altitude:
#             print(f"[FLIGHT_CTL] Altitude: {altitude}")

#         self.cmd.altitude = altitude
