from dataclasses import dataclass, field
from enum import Enum
from math import pi

import numpy as np
import cv2

from common import Context
from debug import export_array
from log import Logger
from navigation import Map, Navigation
from utils import Coords, Timer, Vec2, clip, normalise_angle, rbf_kernel

TWO_PI = 2 * pi
HALF_PI = 0.5 * pi

BOOT_TICKS = 8
SPIN_UP_TICKS = 64
PATH_COMPUTE_TICKS = 64

ERROR_ALTITUDE = 0.05
ERROR_DISTANCE = 0.05
ERROR_WAYPOINT = 0.25
ERROR_PAD_DETECT = 0.01

BOOT_ALTITUDE = 0.5
CRUISING_ALTITUDE = 0.3
SEARCH_ALTITUDE = 0.06
PAD_HEIGHT = 0.1
PAD_ALTITUDE = CRUISING_ALTITUDE - PAD_HEIGHT

COLLISION_RANGE = 0.5
COLLISION_VELOCITY = 0.5
LIMIT_VELOCITY = 0.3
LIMIT_YAW = 1.0
YAW_SPEED_MULTIPLIER = 30.0
YAW_SCAN_RATE = 1.0
YAW_SCAN_RATE_SLOW = 0.25

EXTRA_OFFSET_MAG = 0.05
SEARCH_LOCATIONS: list[Vec2] = [Vec2(4.25, 1.5), Vec2(4.25, 2.25), Vec2(4.25, 0.75)]


class Stage(Enum):
    Boot = 0
    SpinUp = 1
    HomeTakeOff = 2
    ToSearchZone = 3
    ScanHigh = 4
    DescendToScanLow = 5
    ScanLow = 6
    FlyToDetection = 7
    GoToPadDetection = 8
    FindBound = 9
    FlyToDestination = 10
    LandDestination = 11
    WaitAtDestination = 12
    TakeOffAgain = 13
    ReturnHome = 14
    LandHome = 15
    Stop = 16


@dataclass
class FlightCommand:
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    yaw_rate: float = 0.0
    altitude: float = 0.0

    def to_list(self):
        return [self.velocity_x, self.velocity_y, self.yaw_rate, self.altitude]


@dataclass
class FlightState:
    mission_complete: bool = False
    high_alt_map: Map | None = None
    home: Vec2 = field(default_factory=Vec2)
    last_range_down: float = 0.0
    obstacle_avoidance: bool = False
    over_pad: bool = False
    pad_location: Vec2 | None = None
    pad_detection: Vec2 | None = None
    pad_bounds: tuple[Vec2, Vec2] = field(default_factory=lambda: (Vec2(), Vec2()))
    path: list[Coords] | None = None
    scan: bool = False
    scan_speed: float = YAW_SCAN_RATE
    stage: Stage = Stage.Boot
    target_position: Vec2 = field(default_factory=Vec2)
    target_yaw: float = 0.0
    target_altitude: float = 0.0
    timer: float = 0.0


class FlightController(Logger):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.nav = Navigation(ctx)
        self.state = FlightState()

        self.timer = Timer(ctx)
        self.timer.reset()

        self.path_timer = Timer(ctx)

    def update(self) -> FlightCommand:
        state = self.state

        self.nav.update()

        self.detect_pad()

        match self.state.stage:
            case Stage.Boot:
                # Apply a larger starting impulse to spin up the rotors faster
                # This will provide a larger starting error to the PID controller
                state.target_altitude = BOOT_ALTITUDE

                # The scene is not immediately randomised, wait a bit first
                if not self.timer.elapsed_ticks(BOOT_TICKS):
                    return self.compute_flight_command()

                state.home = self.get_position()
                state.target_position = state.home

                self.timer.reset()
                self.transition(Stage.SpinUp)
                return self.update()

            case Stage.SpinUp:
                if not self.timer.elapsed_ticks(SPIN_UP_TICKS):
                    return self.compute_flight_command()

                state.target_altitude = CRUISING_ALTITUDE
                self.transition(Stage.HomeTakeOff)
                return self.update()

            case Stage.HomeTakeOff:
                if not self.is_near_target_altitude():
                    return self.compute_flight_command()

                self.transition(Stage.ToSearchZone)

                state.scan = True
                state.target_position = SEARCH_LOCATIONS[0]
                state.obstacle_avoidance = True

                return self.update()

            case Stage.ToSearchZone:
                if not self.is_near_target():
                    return self.compute_flight_command()

                self.state.scan_speed = YAW_SCAN_RATE_SLOW
                self.timer.reset()
                self.transition(Stage.ScanHigh)
                return self.compute_flight_command()

            case Stage.ScanHigh:
                if not self.timer.elapsed_ticks(64):
                    return self.compute_flight_command()

                self.state.high_alt_map = self.nav.save()
                self.timer.reset()
                self.transition(Stage.DescendToScanLow)
                return self.update()

            case Stage.DescendToScanLow:
                if self.timer.elapsed_ticks(8):
                    self.timer.reset()
                    state.target_altitude -= 0.01

                if state.target_altitude <= SEARCH_ALTITUDE:
                    state.target_altitude = SEARCH_ALTITUDE
                    self.timer.reset()
                    self.transition(Stage.ScanLow)
                    return self.update()

                return self.compute_flight_command()

            case Stage.ScanLow:
                assert state.high_alt_map is not None

                if not self.timer.elapsed_ticks(64):
                    return self.compute_flight_command()

                pad = self.compare_maps(state.high_alt_map, self.nav.save())
                self.info(f"Found pad at {pad}")

                state.target_position = pad
                state.target_altitude = CRUISING_ALTITUDE
                self.transition(Stage.FlyToDetection)
                return self.update()

            case Stage.FlyToDetection:
                if state.over_pad:
                    position = self.get_position()
                    direction = state.target_position - position

                    state.pad_detection = position + direction.set_mag(EXTRA_OFFSET_MAG)
                    state.target_position = state.pad_detection

                    self.transition(Stage.GoToPadDetection)
                    return self.update()

                if not self.is_near_target(ERROR_PAD_DETECT):
                    return self.compute_flight_command()

                self.transition(Stage.Stop)
                return self.update()

            case Stage.GoToPadDetection:
                offset = self.next_bound_offset()

                if offset is None:
                    self.info("All bounds found")
                    (a, b) = state.pad_bounds

                    state.pad_location = 0.5 * (a + b)
                    self.info(f"Pad location: {state.pad_location}")
                    state.target_position = state.pad_location
                    state.scan = False

                    self.transition(Stage.FlyToDestination)
                    return self.update()

                if not self.is_near_target(ERROR_PAD_DETECT):
                    return self.compute_flight_command()

                assert state.pad_detection is not None
                state.target_position = state.pad_detection + offset

                self.transition(Stage.FindBound)
                return self.update()

            case Stage.FindBound:
                if self.is_near_target():
                    self.error("Could not find bound!")
                    self.transition(Stage.Stop)
                    return self.update()

                if not state.over_pad:
                    self.info("Found bound!")
                    self.update_bound(self.get_position())
                    self.info("Bounds are now {}".format(state.pad_bounds))
                    self.info(self.get_bound_direct_sides())

                    assert state.pad_detection is not None
                    state.target_position = state.pad_detection

                    self.transition(Stage.GoToPadDetection)
                    return self.update()

                return self.compute_flight_command()

            case Stage.FlyToDestination:
                if self.is_near_target(ERROR_PAD_DETECT):
                    state.target_altitude = 0.0
                    self.transition(Stage.LandDestination)
                    return self.update()

                return self.compute_flight_command()

            case Stage.LandDestination:
                if self.is_grounded():
                    self.timer.reset()
                    self.transition(Stage.WaitAtDestination)
                    return self.update()

                return self.compute_flight_command()

            case Stage.WaitAtDestination:
                if self.timer.elapsed_ticks(128):
                    state.target_altitude = CRUISING_ALTITUDE
                    self.transition(Stage.TakeOffAgain)
                    return self.update()

                return FlightCommand()

            case Stage.TakeOffAgain:
                if not self.is_near_target_altitude():
                    return self.compute_flight_command()

                state.scan = True
                state.target_position = state.home
                self.transition(Stage.ReturnHome)
                return self.update()

            case Stage.ReturnHome:
                if self.is_near_target():
                    state.scan = False

                    if self.is_facing_target():
                        state.target_altitude = 0.0
                        self.transition(Stage.LandHome)
                        return self.update()

                return self.compute_flight_command()

            case Stage.LandHome:
                if self.is_grounded():
                    self.transition(Stage.Stop)
                    return self.update()

                return self.compute_flight_command()

            case Stage.Stop:
                return FlightCommand()

    def transition(self, stage: Stage) -> None:
        self.info(f"Transitioning to {Stage(stage)}")
        self.state.stage = stage

    # == Sensors == #

    def detect_pad(self) -> None:
        state = self.state
        sensors = self.ctx.sensors

        altitude_delta = sensors.range_down - state.last_range_down
        threshold = PAD_HEIGHT / 2

        if altitude_delta < -threshold:
            self.info("Pad detected")
            state.over_pad = True

        elif altitude_delta > threshold:
            self.info("Leaving pad")
            state.over_pad = False

        state.last_range_down = sensors.range_down

    def is_grounded(self) -> bool:
        return self.ctx.sensors.range_down <= ERROR_ALTITUDE

    def is_near_target_altitude(self) -> bool:
        return self.is_near_altitude(self.state.target_altitude)

    def is_near_altitude(self, altitude: float) -> bool:
        current = self.get_altitude()
        return altitude - ERROR_ALTITUDE <= current <= altitude + ERROR_ALTITUDE

    def get_altitude(self) -> float:
        altitude = self.ctx.sensors.range_down

        if self.state.over_pad:
            altitude += PAD_HEIGHT

        return altitude

    def is_facing_target(self) -> bool:
        return self.is_facing(self.state.target_yaw)

    def is_facing(self, heading: float) -> bool:
        yaw = self.ctx.sensors.yaw
        return heading - ERROR_DISTANCE <= yaw <= heading + ERROR_DISTANCE

    def is_near_target(self, error: float | None = None):
        return self.is_near_position(self.state.target_position, error)

    def is_near_position(self, position: Vec2, error: float | None) -> bool:
        current = self.get_position()
        delta = current - position
        return delta.mag() <= (error or ERROR_DISTANCE)

    def get_position(self) -> Vec2:
        sensors = self.ctx.sensors
        return Vec2(sensors.x_global, sensors.y_global)

    def distance_to_target(self) -> float:
        return (self.get_position() - self.state.target_position).mag()

    # == Search == #

    def compare_maps(self, before: Map, after: Map) -> Vec2:
        before = np.clip(before, 0, 1)
        after = np.clip(after, 0, 1)

        export_array("before", before, cmap="gray")
        export_array("after", after, cmap="gray")

        diff = np.absolute(cv2.subtract(after, before)).astype(np.uint8)
        export_array("diff", diff, cmap="gray")

        diff = cv2.filter2D(diff, -1, rbf_kernel(5, 1.0))
        export_array("diff_blur", diff, cmap="gray")

        max = np.argmax(diff, axis=None)
        (x, y) = np.unravel_index(max, diff.shape)
        return self.nav.to_position((int(x), int(y)))

    def update_bound(self, position: Vec2):
        match self.next_bound_side():
            case 0:
                self.state.pad_bounds[1].x = position.x
            case 1:
                self.state.pad_bounds[0].y = position.y
            case 2:
                self.state.pad_bounds[0].x = position.x
            case 3:
                self.state.pad_bounds[1].y = position.y

    def next_bound_offset(self) -> Vec2 | None:
        match self.next_bound_side():
            case 0:
                return Vec2(1.0, 0.0)
            case 1:
                return Vec2(0.0, 1.0)
            case 2:
                return Vec2(-1.0, 0.0)
            case 3:
                return Vec2(0.0, -1.0)
            case None:
                return None

    def next_bound_side(self) -> int | None:
        try:
            return list(self.get_bound_direct_sides()).index(0.0)
        except ValueError:
            return None

    def get_bound_direct_sides(self):
        (a, b) = self.state.pad_bounds
        return (b.x, a.y, a.x, b.y)

    # == Control == #

    def compute_flight_command(self):
        sensors = self.ctx.sensors
        state = self.state

        position = self.get_position()

        start = self.nav.to_coords(position)
        end = self.nav.to_coords(state.target_position)

        if self.path_timer.elapsed_ticks(PATH_COMPUTE_TICKS):
            self.path_timer.reset()
            state.path = self.nav.compute_path(start, end)

        path_to_broadcast = [[x, y] for (x, y) in state.path or []]
        self.ctx.outlet.broadcast({"type": "path", "data": path_to_broadcast})

        next_target = self.get_next_waypoint()

        yaw = sensors.yaw
        yaw_rate = YAW_SCAN_RATE
        heading_error = normalise_angle(state.target_yaw - yaw)

        position_error = next_target - position
        velocity = position_error.rotate(-yaw)
        velocity = velocity.set_mag(self.distance_to_target())
        velocity = velocity.limit_mag(LIMIT_VELOCITY)

        altitude = state.target_altitude

        if state.over_pad:
            altitude -= PAD_HEIGHT

        if not self.state.scan:
            yaw_rate = clip(heading_error, -LIMIT_YAW, LIMIT_YAW)

        cmd = FlightCommand(
            velocity_x=velocity.x,
            velocity_y=velocity.y,
            altitude=altitude,
            yaw_rate=yaw_rate,
        )

        # return self.apply_collision_avoidance(cmd)
        return cmd

    def get_next_waypoint(self) -> Vec2:
        if self.state.path is None:
            return self.state.target_position

        while self.close_to_next_waypoint():
            if len(self.state.path) == 0:
                return self.state.target_position

            self.state.path.pop(0)

        return self.nav.to_position(self.state.path[0])

    def close_to_next_waypoint(self) -> bool:
        if self.state.path is None or len(self.state.path) == 0:
            return True

        position = self.get_position()
        target = self.nav.to_position(self.state.path[0])
        return (position - target).mag() <= ERROR_WAYPOINT

    def apply_collision_avoidance(self, cmd: FlightCommand) -> FlightCommand:
        sensors = self.ctx.sensors

        if sensors.range_front <= COLLISION_RANGE:
            cmd.velocity_x = sensors.range_back - COLLISION_RANGE

        if sensors.range_left <= COLLISION_RANGE:
            cmd.velocity_y = sensors.range_left - COLLISION_RANGE

        if sensors.range_right <= COLLISION_RANGE:
            cmd.velocity_y = COLLISION_RANGE - sensors.range_right

        if sensors.range_back <= COLLISION_RANGE:
            cmd.velocity_x = COLLISION_RANGE - sensors.range_back

        return cmd
