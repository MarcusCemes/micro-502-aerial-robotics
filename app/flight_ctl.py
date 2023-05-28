import numpy as np
from loguru import logger

from cflib.positioning.motion_commander import MotionCommander

from .common import Context
from .config import (
    ANGULAR_SCAN_VELOCITY_DEG,
    ANGULAR_VELOCITY_LIMIT_DEG,
    MAX_SLOPE,
    VELOCITY_LIMIT,
    PAD_HEIGHT,
    VERTICAL_VELOCITY_LIMIT,
    VELOCITY_LIMIT_SLOW,
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
        self.z_hist = np.zeros(3)
        self.range_down_list = np.zeros(500)

    def update(self) -> bool:
        return self.next()

    def next(self) -> bool:
        range_down = self._fctx.ctx.sensors.z
        self._fctx.z_hist = np.append(self._fctx.z_hist[1:], range_down)
        next = self._state.next(self._fctx)

        z_hist = self._fctx.ctx.sensors.z
        self.range_down_list = np.append(self.range_down_list[1:], z_hist)

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

    def apply_flight_command(self, mctl: MotionCommander) -> None:
        nav = self._fctx.navigation

        s = self._fctx.ctx.sensors
        t = self._fctx.trajectory

        position = Vec2(s.x, s.y)

        pos_coords = nav.to_coords(position)
        target_coords = nav.to_coords(t.position)

        if self._fctx.ctx.debug_tick:
            path = nav.compute_path(pos_coords, target_coords)

            if path is not None:
                self._fctx.path = [self._fctx.navigation.to_position(c) for c in path]
            else:
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

        if self._fctx.ctx.drone.slow_speed:
            v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT_SLOW)
        else:
            v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT)

        delta_alt = t.altitude - s.z

        if self._fctx.over_pad:
            delta_alt -= PAD_HEIGHT

        vz = clip(delta_alt, -VERTICAL_VELOCITY_LIMIT, VERTICAL_VELOCITY_LIMIT)

        va = ANGULAR_SCAN_VELOCITY_DEG

        if not self._fctx.scan:
            va = clip(
                rad_to_deg(normalise_angle(s.yaw - t.orientation)),
                -ANGULAR_VELOCITY_LIMIT_DEG,
                ANGULAR_VELOCITY_LIMIT_DEG,
            )

        if t.touch_down:
            mctl.land()
            # regarder pour mettre la pos a 0 0 quand il décole pour qu'il reste fixe
            mctl.take_off()
            t.touch_down = False

        # if self._fctx.ctx.debug_tick:
        #     logger.debug(
        #         f"p: {position}, z: {s.z:.2f} t: {t.position}, v: {v}, vz: {vz:.2f}"
        #     )

        mctl.start_linear_motion(v.x, v.y, vz, va)
