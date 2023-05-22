from loguru import logger

from cflib.positioning.motion_commander import MotionCommander

from .common import Context
from .config import (
    ANGULAR_SCAN_VELOCITY_DEG,
    ANGULAR_VELOCITY_LIMIT_DEG,
    PAD_THRESHOLD,
    VELOCITY_LIMIT,
    VERTICAL_VELOCITY_LIMIT,
)
from .flight_states import Boot, FlightContext, State, Stop
from .navigation import Navigation
from .utils.math import Vec2, clip, deg_to_rad, normalise_angle, rad_to_deg


class FlightController:
    _state: State

    def __init__(self, ctx: Context, navigation: Navigation) -> None:
        self._state = Boot()

        self._fctx = FlightContext(
            ctx,
            navigation,
        )

        self._last_altitude = 0.0

    def update(self) -> bool:
        self.detect_pad()
        return self.next()

    def next(self) -> bool:
        next = self._state.next(self._fctx)

        if next is not None:
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

        path = nav.compute_path(pos_coords, target_coords)

        next_waypoint = target_coords
        if path is not None and len(path) >= 1:
            next_waypoint = path[0]

        next_location = nav.to_position(next_waypoint).set_mag(0.5)
        v = (next_location - position).rotate(-deg_to_rad(s.yaw)).limit(VELOCITY_LIMIT)

        vz = clip(t.altitude - s.z, -VERTICAL_VELOCITY_LIMIT, VERTICAL_VELOCITY_LIMIT)

        va = ANGULAR_SCAN_VELOCITY_DEG

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
        delta = self._last_altitude - self._fctx.ctx.sensors.z

        if delta > PAD_THRESHOLD:
            logger.info(f"🎯 Detected pad!")
            self._fctx.over_pad = True
        elif delta < PAD_THRESHOLD:
            logger.info(f"🎯 Lost pad!")
            self._fctx.over_pad = False

        self._last_altitude = self._fctx.ctx.sensors.z

        self._last_altitude = self._fctx.ctx.sensors.z
