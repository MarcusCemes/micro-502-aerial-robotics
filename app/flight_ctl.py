import numpy as np
from loguru import logger

from cflib.positioning.motion_commander import MotionCommander

from .common import Context
from .config import (
    ANGULAR_SCAN_VELOCITY_DEG,
    ANGULAR_VELOCITY_LIMIT_DEG,
    VELOCITY_LIMIT,
    VELOCITY_LIMIT_FAST,
)
from .flight_states import Boot, FlightContext, State, Stop
from .navigation import Navigation
from .utils.math import Vec2, clip, normalise_angle, rad_to_deg


class FlightController:
    """
    The FlightController controls the finite-state machine that is used to
    control the actions of the drone. The FSM is evaluated once new sensor
    data is received, updating the desired Trajectory in FlightContext
    which is then used to generate the correct flight command to transmit
    to the drone.
    """

    _state: State

    def __init__(self, ctx: Context, navigation: Navigation) -> None:
        self._state = Boot()

        self._fctx = FlightContext(ctx, navigation)

    def update(self) -> bool:
        """
        Evalutes the finite-state machine recursively until a stable State
        class is returned. If the Stop state is returned, the function returns
        `True` to inducate that the FSM has terminated.
        """

        next = self._state.next(self._fctx)

        if next is not None:

            # A state is not allowed to return itself as the next state,
            # which is likely a cause of infinite recursion.
            if type(next) == self._state:
                logger.error("🚨 Infinite loop detected in state machine")
                return True

            logger.info(f"🚩 Transition to state {next.__class__.__name__}")
            self._state = next

            next.start(self._fctx)

            return self.update()

        return type(self._state) == Stop

    def apply_flight_command(self) -> None:
        """
        This method is called at the end of the update loop to generate the
        correct drone flight command (hover setpoint) once the Trajectory
        dataclass has been updated.
        """

        nav = self._fctx.navigation

        s = self._fctx.ctx.sensors
        t = self._fctx.trajectory

        position = Vec2(s.x, s.y)

        pos_coords = nav.to_coords(position)
        target_coords = nav.to_coords(t.position)

        # Compute the path using the obstacle map
        path = None

        if self._fctx.enable_path_finding:
            path = nav.compute_path(pos_coords, target_coords)

        if path is not None:
            self._fctx.path = [self._fctx.navigation.to_position(c) for c in path]

        elif path == []:
            self._fctx.path = None

        # Select the next waypoint as the first element in the path,
        # or the trajectory target if a path is not available.
        while (
            self._fctx.is_near_next_waypoint()
            and self._fctx.path is not None
            and len(self._fctx.path) >= 0
        ):
            self._fctx.path.pop(0)

        next_waypoint = t.position

        if self._fctx.path is not None and len(self._fctx.path) > 0:
            next_waypoint = self._fctx.path[0]

        if self._fctx.ctx.drone.fast_speed:
            v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT_FAST)
        else:
            v = (next_waypoint - position).rotate((-s.yaw)).limit(VELOCITY_LIMIT)

        # Compute the required angular rotation to reach a specific orientation,
        # or use a constant angular speed if scanning is enabled.
        va = ANGULAR_SCAN_VELOCITY_DEG

        if not self._fctx.scan:
            va = clip(
                rad_to_deg(normalise_angle(s.yaw - t.orientation)),
                -ANGULAR_VELOCITY_LIMIT_DEG,
                ANGULAR_VELOCITY_LIMIT_DEG,
            )

        commander = self._fctx.ctx.drone.cf.commander
        commander.send_hover_setpoint(v.x, v.y, va, t.altitude)
