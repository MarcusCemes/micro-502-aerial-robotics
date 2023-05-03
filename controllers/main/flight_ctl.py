from dataclasses import dataclass, field
from enum import Enum
from math import pi

from common import Context
from log import Logger
from navigation import Navigation
from utils import Timer, Vec2, clip, normalise_angle

TWO_PI = 2 * pi
HALF_PI = 0.5 * pi

BOOT_TICKS = 4
SPIN_UP_TIME = 0.5

ERROR_ALTITUDE = 0.05
ERROR_DISTANCE = 0.05

BOOT_ALTITUDE = 0.5
CRUISING_ALTITUDE = 0.3
SEARCH_ALTITUDE = 0.08
PAD_HEIGHT = 0.1
PAD_ALTITUDE = CRUISING_ALTITUDE - PAD_HEIGHT

COLLISION_RANGE = 0.5
COLLISION_VELOCITY = 0.5
LIMIT_VELOCITY = 0.5
LIMIT_YAW = 1.0
YAW_SPEED_MULTIPLIER = 30.0
YAW_SCAN_RATE = 1.0


SEARCH_LOCATIONS: list[Vec2] = [Vec2(4.25, 1.5), Vec2(4.25, 2.25), Vec2(4.25, 0.75)]


class Stage(Enum):
    Boot = 0
    SpinUp = 1
    HomeTakeOff = 2
    ToSearchZone = 3
    DescendToSearch = 4
    Search = 5
    ReturnHome = 6
    Land = 7
    Stop = 8


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
    home: Vec2 = field(default_factory=Vec2)
    last_range_down: float = 0.0
    obstacle_avoidance: bool = False
    over_pad: bool = False
    scan: bool = False
    stage: Stage = Stage.Boot
    target_position: Vec2 = field(default_factory=Vec2)
    target_yaw: float = 0.0
    target_altitude: float = 0.0
    timer: float = 0.0
    update_nav: bool = True


class FlightController(Logger, Timer):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.nav = Navigation(ctx)
        self.state = FlightState()

        self.boot_ticks = 4

    def update(self) -> FlightCommand:
        state = self.state

        if state.update_nav:
            self.nav.update()

        self.detect_pad()

        match self.state.stage:
            case Stage.Boot:
                # Apply a larger starting impulse to spin up the rotors faster
                # This will provide a larger starting error to the PID controller
                state.target_altitude = BOOT_ALTITUDE

                # The scene is not immediately randomised, wait a bit first
                self.boot_ticks -= 1
                if self.boot_ticks > 0:
                    return self.compute_flight_command()

                state.home = self.get_position()
                state.target_position = state.home
                print(state.home)

                self.timer_reset()
                self.transition(Stage.SpinUp)
                return self.update()

            case Stage.SpinUp:
                if not self.timer_elapsed(SPIN_UP_TIME):
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

                self.timer_reset()
                self.transition(Stage.DescendToSearch)
                return self.compute_flight_command()

            case Stage.DescendToSearch:
                if self.timer_elapsed(0.1):
                    self.timer_reset()
                    state.target_altitude -= 0.01

                if state.target_altitude <= SEARCH_ALTITUDE:
                    state.target_altitude = SEARCH_ALTITUDE
                    self.transition(Stage.Search)
                    return self.update()

                return self.compute_flight_command()

            case Stage.Search:
                if not self.is_near_altitude(SEARCH_ALTITUDE):
                    return self.compute_flight_command()

                return self.compute_flight_command()

            # case Stage.ReturnHome:
            #     ctx = self.ctx
            #     sensors = ctx.sensors
            #     state = ctx.state

            #     self.detect_pad()

            #     position = Vec2(sensors.x_global, sensors.y_global)
            #     delta = self.home - position

            #     if delta.mag() <= ERROR_DISTANCE:
            #         self.info("Arrived home, landing")
            #         self.stage = Stage.Land
            #         return self.update()

            #     altitude = PAD_ALTITUDE if state.over_pad else CRUISING_ALTITUDE
            #     cmd = self.generate_command(delta, altitude)
            #     return self.apply_collision_avoidance(cmd)

            # case Stage.Land:
            #     if self.ctx.sensors.range_down <= ERROR_ALTITUDE:
            #         if self.first_flight:
            #             self.info("First flight complete, taking off again")
            #             self.first_flight = False
            #             self.ctx.state.target = Vec2(*LOC_START)
            #             self.stage = Stage.HomeTakeOff
            #             return self.update()

            #         self.info("Mission complete, stopping")
            #         self.stage = Stage.Stop

            #     return FlightCommand()

            # case Stage.Stop:
            #     return FlightCommand()

            case _:
                return self.compute_flight_command()

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

    def is_near_target_altitude(self) -> bool:
        return self.is_near_altitude(self.state.target_altitude)

    def is_near_altitude(self, altitude: float) -> bool:
        current = self.get_altitude()
        return altitude - ERROR_ALTITUDE <= current <= altitude + ERROR_ALTITUDE

    def get_altitude(self) -> float:
        altitude = self.ctx.sensors.range_down
        offset = PAD_ALTITUDE if self.state.over_pad else 0
        return altitude + offset

    def is_near_target(self):
        return self.is_near_position(self.state.target_position)

    def is_near_position(self, position: Vec2) -> bool:
        current = self.get_position()
        delta = current - position
        return delta.mag() <= ERROR_DISTANCE

    def get_position(self) -> Vec2:
        sensors = self.ctx.sensors
        return Vec2(sensors.x_global, sensors.y_global)

    # == Control == #

    def compute_flight_command(self):
        sensors = self.ctx.sensors
        state = self.state

        position = self.get_position()
        yaw = sensors.yaw

        position_error = state.target_position - position
        heading_error = normalise_angle(state.target_yaw - sensors.yaw)

        yaw_rate = YAW_SCAN_RATE
        velocity = position_error.rotate(-yaw).limit_mag(LIMIT_VELOCITY)

        alt_offset = PAD_ALTITUDE if state.over_pad else 0
        altitude = state.target_altitude - alt_offset

        if not self.state.scan:
            yaw_rate = clip(heading_error, -LIMIT_YAW, LIMIT_YAW)

        cmd = FlightCommand(
            velocity_x=velocity.x,
            velocity_y=velocity.y,
            altitude=altitude,
            yaw_rate=yaw_rate,
        )

        # if self.ctx.debug_tick:
        #     print(cmd)

        return self.apply_collision_avoidance(cmd)

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
