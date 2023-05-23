import numpy as np
from loguru import logger

from cflib.positioning.motion_commander import MotionCommander

from .common import Context
from .config import (
    ANGULAR_SCAN_VELOCITY_DEG,
    ANGULAR_VELOCITY_LIMIT_DEG,
    MAX_SLOPE,
    VELOCITY_LIMIT,
    VERTICAL_VELOCITY_LIMIT,
)
from .flight_states import Boot, FlightContext, State, Stop
from .navigation import Navigation
from .utils.math import Vec2, clip, normalise_angle, rad_to_deg


class FlightController:
    _state: State

    def __init__(self, ctx: Context, navigation: Navigation) -> None:
        self._state = Boot()

        self._fctx = FlightContext(
            ctx,
            navigation,
        )
        self.z_hist = np.zeros(5)

    def update(self) -> bool:
        self.detect_pad()
        return self.next()

    def next(self) -> bool:
        next = self._state.next(self._fctx)

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
            logger.info("🚧 No path found, going straight to target")

        # logger.debug(f"path {path}")
        # if path is not None and len(path) >= 1:
        #     next_waypoint = path[0]
        logger.debug(f"current coord {position}")
        logger.debug(f"next_waypoint {next_waypoint}")
        # next_location = nav.to_position(next_waypoint).set_mag(0.5)

        v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT)

        vz = clip(t.altitude - s.z, -VERTICAL_VELOCITY_LIMIT, VERTICAL_VELOCITY_LIMIT)

        va = ANGULAR_SCAN_VELOCITY_DEG

        if self._fctx.ctx.debug_tick:
            logger.debug(f"At {position}")
            logger.debug(f"Next waypoint: {next_waypoint}")
            logger.debug(f"Velocity: {v}")

        if not self._fctx.scan:
            va = clip(
                rad_to_deg(normalise_angle(t.orientation - s.yaw)),
                -ANGULAR_VELOCITY_LIMIT_DEG,
                ANGULAR_VELOCITY_LIMIT_DEG,
            )

        if self._fctx.ctx.debug_tick:
            logger.debug(
                f"p: {position}, z: {s.z:.2f} t: {t.position}, v: {v}, vz: {vz:.2f}"
            )

        mctl.start_linear_motion(v.x, v.y, vz, va)

    def detect_pad(self) -> None:
        z_hist = self._fctx.ctx.sensors.z
        self.z_hist = np.append(self.z_hist[1:], z_hist)
        slope, _ = np.polyfit(np.arange(5), self.z_hist, 1)
        if np.abs(slope) > MAX_SLOPE:
            logger.info(f"🎯 Detected pad!")
            logger.info(f"🍆👉👌💦❤")
            self._fctx.over_pad = True
        elif np.abs(slope) < -MAX_SLOPE and self._fctx.over_pad:
            # risque de poser pbm: slope hard négative quand on arrive sur le pad, puis positive une fois que le shift sort du vecteur, à tester
            logger.info(f"🎯 Lost pad!")
            self._fctx.over_pad = False

        # delta = np.abs(self._last_altitude - self._fctx.ctx.sensors.z)

        # if delta > PAD_THRESHOLD:
        #     logger.info(f"🎯 Detected pad!")
        #     self._fctx.over_pad = True
        # elif delta < PAD_THRESHOLD:
        #     logger.info(f"🎯 Lost pad!")
        #     self._fctx.over_pad = False
