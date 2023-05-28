import numpy as np
from loguru import logger

from cflib.positioning.motion_commander import MotionCommander

from .common import Context
from .config import (
    ANGULAR_SCAN_VELOCITY_DEG,
    ANGULAR_VELOCITY_LIMIT_DEG,
    VELOCITY_LIMIT,
    PAD_HEIGHT,
    VERTICAL_VELOCITY_LIMIT,
    VELOCITY_LIMIT_SLOW,
    VELOCITY_LIMIT_FAST,
)
from .flight_states import Boot, FlightContext, State, Stop
from .navigation import Navigation
from .utils.math import Vec2, clip, normalise_angle, rad_to_deg

import matplotlib.pyplot as plt


class FlightController:
    _state: State

    def __init__(self, ctx: Context, navigation: Navigation) -> None:
        self._state = Boot()

        self._fctx = FlightContext(ctx, navigation)

        self.range_down_list = np.zeros(500)

    def update(self) -> bool:
        return self.next()

    def next(self) -> bool:
        next = self._state.next(self._fctx)

        if self._fctx.ctx.debug_tick:
            plt.figure("Range down")
            plt.plot(np.arange(len(self.range_down_list)), self.range_down_list)
            plt.savefig("output/range_down.png")

        if next is not None:
            if type(next) == self._state:
                logger.error("🚨 Infinite loop detected in state machine")
                return True

            logger.info(f"🎲 Transition to state {next.__class__.__name__}")
            self._state = next

            next.start(self._fctx)
            return self.update()

        return type(self._state) == Stop

    def apply_flight_command(self) -> None:
        # start_time = time()

        nav = self._fctx.navigation

        s = self._fctx.ctx.sensors
        t = self._fctx.trajectory

        position = Vec2(s.x, s.y)

        pos_coords = nav.to_coords(position)
        target_coords = nav.to_coords(t.position)

        path = nav.compute_path(pos_coords, target_coords)

        if path is not None:
            self._fctx.path = [self._fctx.navigation.to_position(c) for c in path]

        elif path == []:
            self._fctx.path = None

        while (
            self._fctx.is_near_next_waypoint()
            and self._fctx.path is not None
            and len(self._fctx.path) >= 0
        ):
            self._fctx.path.pop(0)

        next_waypoint = t.position

        if self._fctx.path is not None and len(self._fctx.path) > 0:
            next_waypoint = self._fctx.path[0]
        else:
            # logger.info("🚧 No path found, going straight to target")
            pass

        # if self._fctx.ctx.drone.slow_speed:
        #     v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT_SLOW)
        if self._fctx.ctx.drone.fast_speed:
            v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT_FAST)
        else:
            v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT)

        target_altitude = t.altitude

        va = ANGULAR_SCAN_VELOCITY_DEG

        if not self._fctx.scan:
            va = clip(
                rad_to_deg(normalise_angle(s.yaw - t.orientation)),
                -ANGULAR_VELOCITY_LIMIT_DEG,
                ANGULAR_VELOCITY_LIMIT_DEG,
            )

        # if self._fctx.ctx.debug_tick:
        #     logger.debug(f"Target altitude is {target_altitude:.2f}")

        self._fctx.ctx.drone.cf.commander.send_hover_setpoint(
            v.x, v.y, va, target_altitude
        )
