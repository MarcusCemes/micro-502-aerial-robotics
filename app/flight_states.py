from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Final

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

import json

from .common import Context
from .config import (
    ALTITUDE_ERROR,
    CRUISING_ALTITUDE,
    INITIAL_POSITION,
    POSITION_ERROR,
    POSITION_ERROR_PAD,
    LINE_TARGET_SEARCH,
    HOME_PAD_ERROR,
    POSITION_ERROR_PAD,
)
from .navigation import Navigation
from .utils.math import Vec2
from .utils.timer import Timer

# == Types == #

@dataclass
class Trajectory:
    """
    Dataclass containing the desired drone trajectory.
    This class is ready by `apply_flight_command()` to generate
    the drone command signal.
    """

    altitude: float = 0.0
    orientation: float = 0.0
    position: Vec2 = field(default_factory=Vec2)
    touch_down: bool = False


@dataclass
class FlightContext:
    """
    Dataclass containing shared state that can be easily shared between
    state classes relating to the FlightController's finite state machine
    evaluation. This class also contains useful shared methods, such as
    `is_near_target()` that compares the drone's position with
    the set trajectory position.

    Contains a reference to the application Context class, the Navigation
    class and flight-related mutable text.
    """

    ctx: Context
    navigation: Navigation

    trajectory: Trajectory = field(default_factory=Trajectory)

    home_pad: Vec2 | None = None
    pad_detection: bool = False
    over_pad: bool = True
    path: list[Vec2] | None = None
    scan: bool = False
    target_pad: Vec2 | None = None
    id = 0
    enable_path_finding: bool = True

    # == Sensors == #

    def is_near_next_waypoint(self, error=POSITION_ERROR) -> bool:
        if self.path is None or len(self.path) == 0:
            return False

        else:
            return self.is_near_position(self.path[0], error)

    def is_near_target(self, error=POSITION_ERROR) -> bool:
        return self.is_near_position(self.trajectory.position, error)

    def has_crossed_the_line(self) -> bool:
        if self.ctx.sensors.x > LINE_TARGET_SEARCH:
            return True

    def is_near_target_pad(self, error=POSITION_ERROR_PAD) -> bool:
        return self.is_near_position(self.trajectory.position, error)

    def is_near_home(self, error=HOME_PAD_ERROR) -> bool:
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
    """
    Interface for the states in the finite state machine. All states
    should ensure that they implement this protocol to be compatible
    with the FSM.

    Requires two methods, `start()` which is called once on the first tick
    and `next()` which is called on every tick of the FSM evaluation.

    Returning a new `State` instance will signal the FSM to transition
    to that state and reavluate the FSM with the new state.
    """
    def start(self, fctx: FlightContext) -> None:
        return

    def next(self, fctx: FlightContext) -> State | None:
        ...


# == States == #

class Boot(State):
    """
    The boot state is the first state of the flight process. It is used to initialize the drone.
    
    It will set the initial position of the drone and the cruising altitude.

    It will then return the takeoff state.
    """
    def next(self, fctx: FlightContext) -> State | None:
        (x, y) = INITIAL_POSITION
        fctx.trajectory.position = Vec2(x, y)
        fctx.home_pad = fctx.get_position()

        return Takeoff()


class Takeoff(State):
    """
    The takeoff state is used to takeoff the drone.
    
    It will set the cruising altitude and return the cruising state, "Cross"
    """
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.altitude = CRUISING_ALTITUDE

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_altitude():
            fctx.pad_detection = True
            return Cross()
        return None


class Cross(State):
    """
    The cross state is used to cross the obstacle field until the cross line is reached.
    
    It will set the cruising altitude and return the Target searching state.
    """
    def start(self, fctx: FlightContext) -> None:
        logger.info("❎ Crossing")
        fctx.trajectory.position = Vec2(4.5, 1.5)
        fctx.scan = True

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.has_crossed_the_line():
            return TargetSearch()

        return None


class TargetSearch(State):
    """
    The target search state is used to search for the target pad.

    The drone is going trough a list of points, avoiding obstacles, doing grid search.

    It will update the probability map to detect fluctuations, representing the target pad edges. 

    Once the target pad is found, it will return the go to target state.
    """
    def __init__(self):
        self.research_points = []
        self.map_offset = 3.5
        self.index = 0

    def start(self, fctx: FlightContext):
        # compute target map
        self.compute_target_map(fctx)
        # set target
        fctx.trajectory.position = fctx.navigation.to_position(
            self.research_points[self.index]
        )

    def next(self, fctx: FlightContext):
        self.update_research_point(fctx)

        # logger.debug(f"Pos {fctx.navigation.global_position()}")

        if self.index >= 1:
            fctx.ctx.drone.prob_map.fill(fctx)
            fctx.scan = False

        if fctx.ctx.drone.prob_map.two_peaks():
            logger.debug("camel found 🐫")
            offset_detection =  Vec2(fctx.ctx.sensors.vx, fctx.ctx.sensors.vy).set_mag(0.08)
            logger.debug(f"velocity offset: {offset_detection}")
            fctx.target_pad = fctx.ctx.drone.prob_map.find_mean_position() - offset_detection
            return GoToTarget()

        if fctx.is_near_target():
            self.index = self.index + 1

            # logger.debug(f"max prob {np.max(fctx.ctx.drone.prob_map.probability_map)}")
            fctx.ctx.drone.prob_map.save()

            if self.index == len(self.research_points):
                fctx.target_pad = fctx.ctx.drone.prob_map.find_mean_position()
                logger.debug(f"target pad {fctx.target_pad}")
                return GoToTarget()
            else:
                fctx.trajectory.position = fctx.navigation.to_position(
                    self.research_points[self.index]
                )

        return None

    def compute_target_map(self, fctx: FlightContext):
        research_points1 = [
            (4.7, 2.7),
            (4.7, 1.9),
            (4.7, 1.1),
            (4.7, 0.3),
            (4.5, 0.3),
            (4.5, 1.1),
            (4.5, 1.9),
            (4.5, 2.7),
            (4.2, 2.7),
            (4.2, 1.9),
            (4.2, 1.1),
            (4.2, 0.3),
            (3.9, 0.3),
            (3.9, 1.1),
            (3.9, 1.9),
            (3.9, 2.7),
            (3.6, 2.7),
            (3.6, 1.9),
            (3.6, 1.1),
            (3.6, 0.3),
        ]

        research_points = []

        for i in range(len(research_points1)):
            point = fctx.navigation.to_coords(
                Vec2(research_points1[i][0], research_points1[i][1])
            )
            if fctx.navigation.coords_in_range(point) and fctx.navigation.is_visitable(
                point
            ):
                research_points.append(point)

        self.research_points = research_points

    def update_research_point(self, fcxt: FlightContext):
        index = 0
        while index < len(self.research_points):
            if not fcxt.navigation.is_visitable(self.research_points[index]):
                logger.info(
                    f"🐣Research point {self.research_points[index]} is not visitable"
                )
                self.research_points.pop(index)
            else:
                index += 1


class GoToTarget(State):
    """
    The GoToTarget state is used to go to the target pad.

    It will set the target pad and return the TouchDown state.
    """
    def start(self, fctx: FlightContext) -> None:
        correction_offset = Vec2(0.0, 0.0)
        fctx.trajectory.position = fctx.target_pad + correction_offset

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_pad():
            return TouchDown()
        

class TouchDown(State):
    """
    
    """
    def start(self, fctx: FlightContext):
        logger.info(f"👌 Touching down")
        self.touched = False
        

    def next(self, fctx: FlightContext) -> State | None:
        fctx.trajectory.altitude -= 0.005
        if fctx.trajectory.altitude <= -0.14 and not self.touched:
            fctx.trajectory.altitude = 0.4
            self.touched = True

        if fctx.is_near_target_altitude() and self.touched:
            return ReturnHome()

        return None


class ReturnHome(State):
    def start(self, fctx: FlightContext):
        assert fctx.home_pad is not None

        fctx.scan = True
        fctx.enable_path_finding = True
        fctx.ctx.drone.slow_speed = False

        logger.info(f"🏠 Returning home to {fctx.home_pad}")
        correction_offset = Vec2(0.0, 0.0)
        fctx.trajectory.position = fctx.home_pad + correction_offset
        # print(f"home pad:  {fctx.home_pad.x}, {fctx.home_pad.y}")
        logger.info(
            f"drone position: {fctx.navigation.global_position().x}, {fctx.navigation.global_position().y}"
        )

    def next(self, fctx: FlightContext) -> State | None:
        if (
            fctx.is_near_target_pad()
        ):  # or (pad_detection(fctx) and fctx.is_near_home()):
            logger.info(
                f"drone position: {fctx.navigation.global_position().x}, {fctx.navigation.global_position().y}"
            )
            fctx.target_pad = fctx.home_pad
            return GoLower()

        return None
    

class GoLower(State):
    def start(self, fctx: FlightContext):
        fctx.trajectory.altitude = 0.1
        fctx.trajectory.orientation = 0.0
        fctx.trajectory.position = fctx.target_pad

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_altitude():
            return Stop()

        return None


class Stop(State):
    def next(self, _) -> State | None:
        return None
