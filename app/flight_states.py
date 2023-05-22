from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from scipy.ndimage import convolve
import math

from .common import Context
from .config import ALTITUDE_ERROR, INITIAL_POSITION, POSITION_ERROR
from .utils.math import Vec2
from .utils.timer import Timer
from .navigation import Navigation
from loguru import logger

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
    navigation: Navigation

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

        (x, y) = INITIAL_POSITION
        fctx.trajectory.position = Vec2(x, y)

        return Takeoff()


class Takeoff(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.altitude = 0.5

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_altitude():
            return GoForward()

        return None


class Scan(State):
    def __init__(self):
        self._timer = Timer()

    def start(self, fctx: FlightContext) -> None:
        fctx.scan = True
        self._timer.reset()

    def next(self, fctx: FlightContext) -> State | None:
        if self._timer.is_elapsed(10.0):
            fctx.scan = False
            return Stop()

        return None


class GoForward(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.position.x = 2.0
        # fctx.trajectory.orientation = pi
        fctx.scan = True

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            return ReturnHome()

        return None


# class GoBack(State):
#     def start(self, fctx: FlightContext) -> None:
#         fctx.trajectory.position = Vec2()

#     def next(self, fctx: FlightContext) -> State | None:
#         if fctx.is_near_target():
#             return Stop()

#         return None


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


class TargetSearch(State):
    def __init__(self):
        self.research_points = []
        self.index = 0

    def start(self, fctx: FlightContext):
        # compute target map
        self.compute_target_map(fctx)
        # set target
        fctx.trajectory.position = self.research_points[self.index]

    def next(self, fctx: FlightContext):
        if self.index == len(self.research_points):
            logger.info("No target found")
            return Stop()
        if fctx.is_near_target():
            # move to next target point
            self.index = self.index + 1
            return TargetSearch()

    def compute_target_map(self, fctx: FlightContext):
        research_points1 = [
            (4.7, 2.7),
            (4.7, 1.9),
            (4.7, 1.1),
            (4.7, 0.3),
            (4.2, 0.3),
            (4.2, 1.1),
            (4.2, 1.9),
            (4.2, 2.7),
            (3.8, 2.7),
            (3.8, 1.9),
            (3.8, 1.1),
            (3.8, 0.3),
        ]

        research_points2 = [
            (4.0, 0.8),
            (4.0, 1.5),
            (4.0, 2.3),
            (4.4, 2.3),
            (4.4, 1.5),
            (4.4, 0.8),
        ]

        research_points3 = [
            (4.7, 0.8),
            (4.7, 1.5),
            (4.7, 2.3),
            (4.4, 2.7),
            (4.4, 1.9),
            (4.4, 1.1),
            (4.4, 0.3),
            (4.2, 0.8),
            (4.2, 1.5),
            (4.2, 2.3),
            (4.0, 2.7),
            (4.0, 1.9),
            (4.0, 1.1),
            (4.0, 0.3),
            (3.8, 0.8),
            (3.8, 1.5),
            (3.8, 2.3),
        ]

        occupancy_grid = fctx.navigation.map.copy()
        kernel = np.ones((9, 9), np.uint8)
        occupancy_grid = convolve(occupancy_grid, kernel)

        i = 0
        while i < len(research_points1):
            point = fctx.navigation.to_coords(
                Vec2([research_points1[i][0], research_points1[i][1]])
            )
            point = ((np.rint(point[0])).astype(int),
                     (np.rint(point[1])).astype(int))
            if occupancy_grid[point]:
                del research_points1[i]
            else:
                i += 1

        i = 0
        while i < len(research_points2):
            point = fctx.navigation.to_coords(
                Vec2([research_points2[i][0], research_points2[i][1]]))
            point = ((np.rint(point[0])).astype(int),
                     (np.rint(point[1])).astype(int))
            if occupancy_grid[point]:
                del research_points2[i]
            else:
                i += 1

        i = 0
        while i < len(research_points3):
            point = fctx.navigation.to_coords(
                Vec2([research_points3[i][0], research_points3[i][1]]))
            point = ((np.rint(point[0])).astype(int),
                     (np.rint(point[1])).astype(int))
            if occupancy_grid[point]:
                del research_points3[i]
            else:
                i += 1

            # Move at the end isolated points
        max_dist = 1.50
        min_neighbourg = 3

        nb = 0
        id = 0
        while nb < len(research_points2):
            given_point = research_points2[id]
            count = 0
            for point in research_points2:
                if self.distance(point, given_point) < max_dist:
                    count += 1

            if count <= min_neighbourg:
                research_points2.remove(given_point)
                research_points2.append(given_point)
            else:
                id += 1
            nb += 1

        nb = 0
        id = 0
        while nb < len(research_points3):
            given_point = research_points3[id]
            count = 0
            for point in research_points3:
                if self.distance(point, given_point) < max_dist:
                    count += 1

            if count <= min_neighbourg:
                print(given_point, ' put at the end')
                research_points3.remove(given_point)
                research_points3.append(given_point)
            else:
                id += 1
            nb += 1

        research_points = research_points1.copy()
        research_points += research_points2
        research_points += research_points3

        self.research_points = research_points

    def distance(self, p1, p2):
        """
        Return the distance between two points 

        Args:
            p1 (Tuple): First point
            p2 (Tuple): Second point

        Returns:
            (Double): Distance between the points 
        """
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


class ReturnHome(State):
    def start(self, fctx: FlightContext):
        assert fctx.home_pad is not None

        # kalman is reset when the motors stop at the top pad
        fctx.trajectory.position = fctx.home_pad

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            return GoLower()

        return None
