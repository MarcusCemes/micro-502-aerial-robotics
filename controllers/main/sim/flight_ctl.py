from dataclasses import dataclass, field
from enum import Enum
from math import pi

import cv2
import numpy as np

from .common import Context
from .debug import export_array
from .log import Logger
from .navigation import Map, Navigation
from .utils import Coords, Timer, Vec2, clip, normalise_angle, rbf_kernel

TWO_PI = 2 * pi
HALF_PI = 0.5 * pi

BOOT_TICKS = 8
SPIN_UP_TICKS = 64
# PATH_COMPUTE_TICKS = 32

ERROR_ALTITUDE = 0.05
ERROR_DISTANCE = 0.05
ERROR_WAYPOINT = 0.25
ERROR_PAD_DETECT = 0.01

BOOT_ALTITUDE = 0.5
CRUISING_ALTITUDE = 0.3
SEARCH_ALTITUDE = 0.06
SCAN_ALTITUDE = 0.2
PAD_HEIGHT = 0.1
PAD_ALTITUDE = CRUISING_ALTITUDE - PAD_HEIGHT
PAD_SIZE = 0.3

COLLISION_RANGE = 0.5
COLLISION_VELOCITY = 0.5
VELOCITY_MULTIPLIER = 1.0
LIMIT_VELOCITY = 0.5
LIMIT_VELOCITY_SLOWER = 0.4
LIMIT_VELOCITY_SLOW = 0.2
LIMIT_YAW = 1.0
YAW_SPEED_MULTIPLIER = 30.0
YAW_SCAN_RATE = 2.0
YAW_SCAN_RATE_SLOW = 0.85

LANDING_VELOCITY = 0.1
LANDING_OFFSET = 0.02

EXTRA_OFFSET_MAG = 0.1

SEARCH_LOCATIONS: list[Vec2] = [
    Vec2(4.35, 1.5),
    Vec2(4.25, 2.25),
    Vec2(4.25, 0.75),
    Vec2(4.75, 1.0),
    Vec2(4.75, 2.0),
    Vec2(4.75, 2.75),
    Vec2(4.75, 0.75),
    Vec2(3.5, 1.5),
    Vec2(3.5, 1.0),
    Vec2(3.5, 2.0),
]


class Stage(Enum):
    Boot = 0
    SpinUp = 1
    HomeTakeOff = 2
    ToSearchZone = 3
    ScanHigh = 4
    DescendToScanLow = 5
    ScanLow = 6
    RegainAltitude = 7
    FlyToDetection = 8
    GoToPadDetection = 9
    FindBound = 10
    FlyToDestination = 11
    LandDestination = 12
    WaitAtDestination = 13
    TakeOffAgain = 14
    ReturnHome = 15
    LandHome = 16
    Stop = 17


class Bound(Enum):
    X = 0
    Y = 1


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
    over_pad: bool = False
    pad_location: Vec2 = field(default_factory=Vec2)
    pad_detection: Vec2 | None = None
    path: list[Coords] | None = None
    scan: bool = True
    scan_speed: float = YAW_SCAN_RATE
    search_locations: list[Vec2] = field(default_factory=list)
    stage: Stage = Stage.Boot
    target_position: Vec2 = field(default_factory=Vec2)
    target_velocity: float = LIMIT_VELOCITY
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

        self.detect_pad_presence()

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
                self.timer.reset()
                self.transition(Stage.HomeTakeOff)
                return self.update()

            case Stage.HomeTakeOff:
                if not self.timer.elapsed_ticks(16):
                    return self.compute_flight_command()

                self.transition(Stage.ToSearchZone)

                self.path_timer.timer_ticks = 0
                state.search_locations = SEARCH_LOCATIONS.copy()
                state.target_position = SEARCH_LOCATIONS[0]

                return self.update()

            case Stage.ToSearchZone:
                coords = self.nav.to_coords(state.target_position)

                if state.over_pad:
                    state.pad_detection = self.get_position()
                    state.target_position = state.pad_detection
                    self.transition(Stage.FlyToDetection)
                    return self.update()

                if not self.nav.is_visitable(coords):
                    try:
                        # Go a bit slower, now that we're close to the target
                        state.target_velocity = LIMIT_VELOCITY_SLOWER

                        self.info("Target obstructed, moving to next location")
                        state.target_position = state.search_locations.pop(0)
                    except IndexError:
                        self.error("Search locations exhausted!")
                        self.transition(Stage.ReturnHome)

                    return self.update()

                if not self.is_near_target():
                    return self.compute_flight_command()

                self.nav.high_sensitivity = True
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
                if state.target_altitude <= SEARCH_ALTITUDE:
                    state.target_altitude = SEARCH_ALTITUDE
                    self.timer.reset()
                    self.transition(Stage.ScanLow)
                    return self.update()

                if self.timer.elapsed_ticks(8):
                    self.timer.reset()
                    delta = self.get_altitude() - SEARCH_ALTITUDE
                    state.target_altitude -= max(0.015, 0.1 * delta)

                return self.compute_flight_command()

            case Stage.ScanLow:
                assert state.high_alt_map is not None

                if not self.timer.elapsed_ticks(64):
                    return self.compute_flight_command()

                coords = self.compare_maps(state.high_alt_map, self.nav.save())
                ping = self.nav.to_position(coords)

                self.info(f"Found pad at {coords} {ping}")
                direction = ping - self.get_position()
                state.pad_detection = ping + direction.set_mag(EXTRA_OFFSET_MAG)

                self.nav.high_sensitivity = False
                state.target_altitude = CRUISING_ALTITUDE
                self.transition(Stage.RegainAltitude)
                return self.update()

            case Stage.RegainAltitude:
                if not self.is_near_target_altitude():
                    return self.compute_flight_command()

                assert state.high_alt_map is not None
                assert state.pad_detection is not None

                self.nav.restore(state.high_alt_map)
                state.target_position = state.pad_detection
                self.transition(Stage.FlyToDetection)
                return self.update()

            case Stage.FlyToDetection:
                if state.over_pad:
                    state.scan = False
                    self.transition(Stage.GoToPadDetection)
                    return self.update()

                coords = self.nav.to_coords(state.target_position)
                if self.nav.is_visitable(coords) and not self.is_near_target(
                    ERROR_PAD_DETECT
                ):
                    return self.compute_flight_command()

                try:
                    self.error("Detection not pad or obstructed")
                    state.target_position = state.search_locations.pop(0)
                    self.transition(Stage.ToSearchZone)
                except IndexError:
                    self.error("Search locations exhausted!")
                    self.transition(Stage.ReturnHome)

                return self.update()

            case Stage.GoToPadDetection:
                offset = self.next_bound_offset()

                if offset is None:
                    self.info(f"Pad determined to be at {state.pad_location}")

                    state.target_position = state.pad_location
                    state.scan = False
                    state.target_velocity = LIMIT_VELOCITY

                    self.transition(Stage.FlyToDestination)
                    return self.update()

                if not self.is_near_target(ERROR_PAD_DETECT):
                    return self.compute_flight_command()

                assert state.pad_detection is not None
                # state.target_position = state.pad_detection + offset
                # state.target_velocity = LIMIT_VELOCITY_SLOW
                # self.transition(Stage.FindBound)

                # Skip the bounds check, just try to land
                state.target_position = state.pad_detection
                state.scan = False
                self.transition(Stage.FlyToDestination)
                return self.update()

            case Stage.FindBound:
                if self.is_near_target():
                    self.error("Could not find bound!")
                    self.transition(Stage.Stop)
                    return self.update()

                if not state.over_pad:
                    self.update_pad_location(self.get_position())

                    assert state.pad_detection is not None
                    state.target_position = state.pad_detection

                    self.transition(Stage.GoToPadDetection)
                    return self.update()

                return self.compute_flight_command()

            case Stage.FlyToDestination:
                if (
                    self.is_near_target()
                    and self.get_velocity().mag() <= LANDING_VELOCITY
                ):
                    state.target_altitude = PAD_HEIGHT - LANDING_OFFSET
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
                if self.timer.elapsed_ticks(2):
                    state.scan = True
                    state.scan_speed = YAW_SCAN_RATE
                    state.target_altitude = CRUISING_ALTITUDE
                    self.transition(Stage.TakeOffAgain)
                    return self.update()

                return FlightCommand()

            case Stage.TakeOffAgain:
                if not self.is_near_target_altitude():
                    return self.compute_flight_command()

                state.target_position = state.home
                state.target_velocity = LIMIT_VELOCITY
                self.transition(Stage.ReturnHome)
                return self.update()

            case Stage.ReturnHome:
                if self.is_near_target():
                    state.scan = False

                    state.target_altitude = PAD_HEIGHT - LANDING_OFFSET
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
        self.info(f"🤖 Transitioning to {Stage(stage)}")
        self.state.stage = stage

    # == Sensors == #

    def detect_pad_presence(self) -> None:
        state = self.state
        sensors = self.ctx.sensors

        altitude_delta = sensors.range_down - state.last_range_down
        threshold = PAD_HEIGHT / 2

        if altitude_delta < -threshold:
            self.info("🎉 Pad detected")
            state.over_pad = True

        elif altitude_delta > threshold:
            self.info("👋 Leaving pad")
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
        ss = self.ctx.sensors
        return Vec2(ss.x_global, ss.y_global)

    def get_velocity(self) -> Vec2:
        ss = self.ctx.sensors
        return Vec2(ss.v_forward, ss.v_left).rotate(ss.yaw)

    def distance_to_target(self) -> float:
        return (self.get_position() - self.state.target_position).mag()

    # == Search == #

    def compare_maps(self, before: Map, after: Map) -> Coords:
        before = np.clip(before, 0, 1)
        after = np.clip(after, 0, 1)

        diff = np.absolute(cv2.subtract(after, before)).astype(np.int32)
        export_array("search_diff", diff, cmap="gray")

        kernel = rbf_kernel(9, 2.0)
        diff = cv2.filter2D(diff, -1, kernel)

        max = np.argmax(diff, axis=None)
        self.info(f"Confidence level: {max}")

        (x, y) = np.unravel_index(max, diff.shape)
        return int(x), int(y)

    def update_pad_location(self, position: Vec2):
        match self.next_bound_side():
            case Bound.X:
                self.state.pad_location.x = position.x + 0.5 * PAD_SIZE
            case Bound.Y:
                self.state.pad_location.y = position.y + 0.5 * PAD_SIZE

    def next_bound_offset(self) -> Vec2 | None:
        match self.next_bound_side():
            case Bound.X:
                return Vec2(-1.0, 0.0)
            case Bound.Y:
                return Vec2(0.0, -1.0)
            case None:
                return None

    def next_bound_side(self) -> Bound | None:
        lo = self.state.pad_location
        return Bound.X if lo.x == 0.0 else Bound.Y if lo.y == 0.0 else None

    # == Control == #

    def compute_flight_command(self):
        sensors = self.ctx.sensors
        state = self.state

        position = self.get_position()

        start = self.nav.to_coords(position)
        end = self.nav.to_coords(state.target_position)

        # if self.path_timer.elapsed_ticks(PATH_COMPUTE_TICKS):
        if self.ctx.debug_tick:
            # self.path_timer.reset()
            state.path = self.nav.compute_path(start, end)

        path_to_broadcast = [[x, y] for (x, y) in state.path or []]
        self.ctx.outlet.broadcast({"type": "path", "data": path_to_broadcast})

        next_target = self.get_next_waypoint()

        yaw = sensors.yaw
        yaw_rate = state.scan_speed
        heading_error = normalise_angle(state.target_yaw - yaw)

        altitude = state.target_altitude
        position_error = next_target - position
        velocity = position_error.rotate(-yaw)
        velocity = velocity.set_mag(VELOCITY_MULTIPLIER * self.distance_to_target())
        velocity = velocity.limit_mag(state.target_velocity)

        # Slow down near obstacles
        # print(f"Distance is {self.nav.distance_to_obstacle(start)}")
        # velocity = velocity.limit_mag(max(0.2, self.nav.distance_to_obstacle(start)))

        if state.over_pad:
            altitude -= PAD_HEIGHT

        no_scan = not self.state.scan or (
            state.over_pad and self.get_altitude() < SCAN_ALTITUDE
        )

        if no_scan:
            yaw_rate = clip(heading_error, -LIMIT_YAW, LIMIT_YAW)

        cmd = FlightCommand(
            velocity_x=velocity.x,
            velocity_y=velocity.y,
            altitude=altitude,
            yaw_rate=yaw_rate,
        )

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
