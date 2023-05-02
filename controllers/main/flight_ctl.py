from dataclasses import dataclass
from enum import Enum
from math import pi

from config import LOC_START
from common import Context
from log import Logger
from utils import clip, Vec2

PI_2 = 0.5 * pi

GROUNDED_THRESHOLD = 0.01
CRUISING_ALTITUDE = 0.5
TAKEOFF_ALTITUDE = 0.2

COLLISION_RANGE = 0.5
SEARCH_THRESHOLD = 0.05
YAW_SPEED_MULTIPLIER = 5.0

VELOCITY_LIMIT = 0.5
YAW_LIMIT = 1.0


class Stage(Enum):
    Boot = 0
    TakeOff = 1
    Search = 2
    Land = 3
    Stop = 4


@dataclass
class FlightCommand:
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    yaw_rate: float = 0.0
    altitude: float = 0.0

    def to_list(self):
        return [self.velocity_x, self.velocity_y, self.yaw_rate, self.altitude]


class FlightController(Logger):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.stage = Stage.Boot

        self.first_flight = True

    def update(self) -> FlightCommand:
        match self.stage:
            case Stage.Boot:
                self.stage = Stage.TakeOff
                return self.update()

            case Stage.TakeOff:
                if self.ctx.sensors.range_down >= TAKEOFF_ALTITUDE:
                    self.info("Takeoff complete")
                    self.stage = Stage.Search
                    return self.update()

                return FlightCommand(altitude=CRUISING_ALTITUDE)

            case Stage.Search:
                ctx = self.ctx
                sensors = ctx.sensors
                state = ctx.state

                position = Vec2(sensors.x_global, sensors.y_global)
                delta = state.target - position

                if delta.mag() <= SEARCH_THRESHOLD:
                    self.info("Search complete, landing")
                    self.stage = Stage.Land
                    return self.update()

                if ctx.debug_tick:
                    self.info(f"Search: {position} -> {state.target}")

                cmd = self.generate_command(delta)
                return self.apply_collision_avoidance(cmd)

            case Stage.Land:
                if self.ctx.sensors.range_down <= GROUNDED_THRESHOLD:
                    if self.first_flight:
                        self.info("First flight complete, taking off again")
                        self.first_flight = False
                        self.ctx.state.target = Vec2(*LOC_START)
                        self.stage = Stage.TakeOff
                        return self.update()

                    self.info("Mission complete, stopping")
                    self.stage = Stage.Stop

                return FlightCommand()

            case Stage.Stop:
                return FlightCommand()

    def apply_collision_avoidance(self, cmd: FlightCommand) -> FlightCommand:
        sensors = self.ctx.sensors

        if sensors.range_front <= COLLISION_RANGE:
            if self.ctx.debug_tick:
                self.warn("Front collision imminent")

            direction = sensors.range_left - sensors.range_right
            cmd.velocity_x = 0.0
            cmd.velocity_y = VELOCITY_LIMIT if direction >= 0.0 else -VELOCITY_LIMIT

        if sensors.range_right <= COLLISION_RANGE:
            if self.ctx.debug_tick:
                self.warn("Right collision imminent")
            cmd.velocity_y = VELOCITY_LIMIT

        if sensors.range_left <= COLLISION_RANGE:
            if self.ctx.debug_tick:
                self.warn("Left collision imminent")

            cmd.velocity_y = -VELOCITY_LIMIT

        return cmd

    def generate_command(
        self, delta: Vec2, altitude=CRUISING_ALTITUDE
    ) -> FlightCommand:
        yaw = self.ctx.sensors.yaw
        yaw_error = delta.angle() - yaw
        if yaw_error > pi:
            yaw_error -= 2 * pi

        yaw_rate = clip(YAW_SPEED_MULTIPLIER * yaw_error, -YAW_LIMIT, YAW_LIMIT)
        velocity = delta.rotate(-yaw).limit_mag(VELOCITY_LIMIT)

        if self.ctx.debug_tick:
            self.info(f"Velocity: {velocity}, yaw rate: {yaw_rate}")

        return FlightCommand(
            velocity_x=velocity.x,
            velocity_y=velocity.y,
            yaw_rate=yaw_rate,
            altitude=altitude,
        )


def easing_curve(x: float) -> float:
    abs_x = abs(x)

    if abs_x <= 0.5:
        return 2 * x
    elif abs_x <= 1:
        return 1 if x > 0 else -1
    else:
        return x
