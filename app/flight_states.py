from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Protocol

from .common import Context
from .config import ALTITUDE_ERROR, POSITION_ERROR
from .utils.math import Vec2

# == Simulation states == #

# Boot = 0
# SpinUp = 1
# HomeTakeOff = 2
# ToSearchZone = 3
# ScanHigh = 4
# DescendToScanLow = 5
# ScanLow = 6
# RegainAltitude = 7
# FlyToDetection = 8
# GoToPadDetection = 9
# FindBound = 10
# FlyToDestination = 11
# LandDestination = 12
# WaitAtDestination = 13
# TakeOffAgain = 14
# ReturnHome = 15
# LandHome = 16
# Stop = 17


# == Types == #


@dataclass
class Trajectory:
    altitude: float = 0.0
    orientation: float = 0.0
    position: Vec2 = field(default_factory=Vec2)


@dataclass
class FlightContext:
    ctx: Context

    trajectory: Trajectory = field(default_factory=Trajectory)

    home_pad: Vec2 | None = None
    over_pad: bool = False
    scan: bool = False
    target_pad: Vec2 | None = None

    # == Sensors == #

    def is_near_target(self, error=POSITION_ERROR) -> bool:
        return self.is_near_position(self.trajectory.position, error)

    def is_near_position(self, position: Vec2, error=POSITION_ERROR) -> bool:
        return (self.get_position() - position).abs() < error

    def is_near_target_altitude(self, error=ALTITUDE_ERROR) -> bool:
        return self.is_near_altitude(self.trajectory.altitude, error)

    def is_near_altitude(self, altitude: float, error=ALTITUDE_ERROR) -> bool:
        return abs(self.ctx.sensors.z - altitude) < error

    def get_position(self) -> Vec2:
        return Vec2(self.ctx.sensors.x, self.ctx.sensors.y)


class State(Protocol):
    def start(self, fctx: FlightContext) -> None:
        return

    def next(self, fctx: FlightContext) -> State | None:
        ...


# == States == #


class Boot(State):
    def next(self, fctx: FlightContext) -> State | None:
        fctx.home_pad = fctx.get_position()
        return Takeoff()


class Takeoff(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.altitude = 1.0

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_altitude():
            return GoLower()

        return None


class GoForward(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.position.x = 1.0
        fctx.trajectory.orientation = pi

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            return GoBack()

        return None


class GoBack(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.position = Vec2()

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            return Stop()

        return None


class GoLower(State):
    def start(self, fctx: FlightContext):
        fctx.trajectory.altitude = 0.1
        fctx.trajectory.orientation = 0.0

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_altitude():
            return Stop()

        return None


class Stop(State):
    def next(self, _) -> State | None:
        return None


# class ReturnHome(State):
#     def __init__(self, trajectory: Trajectory):
#         # kalman is reset when the motors stop at the top pad
#         self.position_home_pad = Vec2(0.0, 0.0)
#         self.position_goal_pad = trajectory.position

#     def compute_home_pos(self):