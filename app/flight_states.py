from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from loguru import logger

from .common import Context
from .config import (
    ALTITUDE_ERROR,
    INITIAL_POSITION,
    POSITION_ERROR,
    OSCILLATIONS_AMPLITUDE,
)
from .navigation import Navigation
from .utils.math import Vec2
from .utils.timer import Timer

# == Simulation states == #:

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
    pad_detection: bool = False
    over_pad: bool = True
    path: list[Vec2] | None = None
    path_finding: bool = False
    scan: bool = False
    target_pad: Vec2 | None = None

    # == Sensors == #

    def is_near_next_waypoint(self, error=POSITION_ERROR) -> bool:
        if self.path is None or len(self.path) == 0:
            return False
        else:
            return self.is_near_position(self.path[0], error)

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

        if fctx.is_near_target_altitude():
            return Cross()

        return None


# class Takeoff(State):
#     def start(self, fctx: FlightContext) -> None:
#         fctx.trajectory.altitude = 0.5

#     def next(self, fctx: FlightContext) -> State | None:
#         if fctx.is_near_target_altitude():
#             fctx.pad_detection = True
#             return Cross()

#         return None


class Cross(State):
    def start(self, fctx: FlightContext) -> None:
        logger.info("🛫🏢🏢 Crossing")
        fctx.trajectory.position.x = 3.5
        fctx.scan = True

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            fctx.pad_detection = True
            return TargetSearch()

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


class TargetSearch(State):
    def __init__(self):
        self.research_points = []
        self.index = 0

    def start(self, fctx: FlightContext):
        fctx.scan = True
        fctx.path_finding = True
        # compute target map
        self.compute_target_map(fctx)
        # set target
        fctx.trajectory.position = fctx.navigation.to_position(
            self.research_points[self.index]
        )

    def next(self, fctx: FlightContext):
        self.update_research_point(fctx)
        if self.index == len(self.research_points):
            logger.info("No target found")
            return Stop()
        if fctx.is_near_target():
            # move to next target point
            print("no Target found on this point")
            self.index = self.index + 1
        fctx.trajectory.position = fctx.navigation.to_position(
            self.research_points[self.index]
        )

        # pad detection
        z_hist = self._fctx.ctx.sensors.z
        self.z_hist = np.append(self.z_hist[1:], z_hist)
        slope, _ = np.polyfit(np.arange(len(self.z_hist)), self.z_hist, 1)

        if self._fctx.ctx.debug_tick:
            logger.debug(f"Over pad {self._fctx.over_pad}")
            logger.debug(f"Slope {slope}")

        if slope > MAX_SLOPE:
            logger.info(f"🎯 Detected pad!")
            self._fctx.over_pad = True
        elif slope < -MAX_SLOPE and self._fctx.over_pad:
            # risque de poser pbm: slope hard négative quand on arrive sur le pad, puis positive une fois que le shift sort du vecteur, à tester
            logger.info(f"🎯 Lost pad!")
            self._fctx.over_pad = False

        if fctx.over_pad:
            return TargetCentering()

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

        research_points = []

        for i in range(len(research_points1)):
            point = fctx.navigation.to_coords(
                Vec2(research_points1[i][0], research_points1[i][1])
            )
            # logger.debug("Point: {}".format(point))
            if fctx.navigation.coords_in_range(point) and fctx.navigation.is_visitable(
                point
            ):
                research_points.append(point)

        for i in range(len(research_points2)):
            point = fctx.navigation.to_coords(
                Vec2(research_points2[i][0], research_points2[i][1])
            )
            # logger.debug("Point: {}".format(point))
            if fctx.navigation.coords_in_range(point) and fctx.navigation.is_visitable(
                point
            ):
                research_points.append(point)

        for i in range(len(research_points3)):
            point = fctx.navigation.to_coords(
                Vec2(research_points3[i][0], research_points3[i][1])
            )
            # logger.debug("Point: {}".format(point))
            if fctx.navigation.coords_in_range(point) and fctx.navigation.is_visitable(
                point
            ):
                research_points.append(point)
        print(research_points)
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


class TargetCentering(State):
    def __init__(self):
        self.target_pad: Vec2 | None
        self.platform_x_found: bool = False
        self.platform_y_found: bool = False
        self.last_over_pad: bool = False

        self.axe_up: int = 0
        self.axe_down: int = 1
        self.research_dir: int = 0

        self.axe_X: int = 0
        self.axe_Y: int = 1
        self.research_axe: int = self.axe_X

        self.research_counter: int = 0
        self.change_axe: int = 0
        self.pad_width: float = 0.15

    def start(self, fctx: FlightContext):
        self.init_pos = fctx.navigation.global_position()

    def next(self, fctx: FlightContext) -> State | None:
        self.centering(fctx)

        if self.platform_x_found and self.platform_y_found:
            fctx.target_pad = self.target_pad
            return GoLower()

        return None

    def set_target(self, fctx: FlightContext):
        if self.research_axe == self.axe_X:
            if self.research_dir == self.axe_up:
                vect = Vec2(OSCILLATIONS_AMPLITUDE, 0)
            else:
                vect = Vec2(-OSCILLATIONS_AMPLITUDE, 0)

        else:
            if self.research_dir == self.axe_up:
                vect = Vec2(0, OSCILLATIONS_AMPLITUDE)
            else:
                vect = Vec2(0, -OSCILLATIONS_AMPLITUDE)

        fctx.trajectory.position = self.init_pos + vect

    def centering(self, fctx: FlightContext):
        """
        Find the center of the platform.

        Args:
            sensor_data (Dictionarie): data sensors of the drone

        Returns:
            commande (List): control commande of the drone
            (Boolean): Center found or not
        """

        self.set_target(fctx)

        if fctx.over_pad is not self.last_over_pad:
            self.update_platform_pos(fctx)
        self.last_over_pad = fctx.over_pad

        if self.change_axe >= 2:
            if self.research_axe == self.axe_X:
                self.research_axe = self.axe_Y
            else:
                self.research_axe = self.axe_X

            self.change_axe = 0
            self.research_counter += 1

        if self.research_counter >= 5:
            logger.info(f"🔒 Stuck !!!")
            # stuck = True

        if self.platform_x_found and self.platform_y_found:
            return

        elif self.research_axe == self.axe_X:
            if self.research_dir == self.axe_up:
                if fctx.is_near_target():
                    self.change_axe += 1
                    self.research_dir = self.axe_down

            else:
                if fctx.is_near_target():
                    self.research_dir = self.axe_up

        elif self.research_axe == self.axe_Y:
            if self.research_dir == self.axe_up:
                if fctx.is_near_target():
                    self.change_axe += 1
                    self.research_dir = self.axe_down

            else:
                if fctx.is_near_target():
                    self.research_dir = self.axe_up

    def update_platform_pos(self, fctx: FlightContext):
        """
        Update the position of the platform from the actual position
        and the direction of the movement.

        Args:
            position (List): actual position at the moment of the call
        """

        angle = -math.atan2(fctx.ctx.sensors.vy, fctx.ctx.sensors.vx)

        # Back left
        if angle >= -7 * np.pi / 8 and angle < -5 * np.pi / 8:
            logger.info(f"↙")
            pass

        # Left
        elif angle >= -5 * np.pi / 8 and angle < -3 * np.pi / 8:
            logger.info(f"⬅")
            if fctx.over_pad:
                self.target_pad = Vec2(
                    self.init_pos.x, self.init_pos.y + self.pad_width
                )
            else:
                self.target_pad = Vec2(
                    self.init_pos.x, self.init_pos.y - self.pad_width
                )
            self.platform_y_found = True
            self.change_axe = 0
            self.research_axe = self.axe_X

        # Front left
        elif angle >= -3 * np.pi / 8 and angle < -np.pi / 8:
            logger.info(f"↖")
            pass

        # Front
        elif angle >= -np.pi / 8 and angle < np.pi / 8:
            logger.info(f"⬆")
            if fctx.over_pad:
                self.target_pad = Vec2(
                    self.init_pos.x + self.pad_width, self.init_pos.y
                )
            else:
                self.target_pad = Vec2(
                    self.init_pos.x - self.pad_width, self.init_pos.y
                )
            self.platform_x_found = True
            self.change_axe = 0
            self.research_axe = self.axe_Y

        # Front right
        elif angle >= np.pi / 8 and angle < 3 * np.pi / 8:
            logger.info(f"↗")
            pass

        # Right
        elif angle >= 3 * np.pi / 8 and angle < 5 * np.pi / 8:
            logger.info(f"➡")
            if fctx.over_pad:
                self.target_pad = Vec2(
                    self.init_pos.x, self.init_pos.y - self.pad_width
                )
            else:
                self.target_pad = Vec2(
                    self.init_pos.x, self.init_pos.y + self.pad_width
                )
            self.platform_y_found = True
            self.change_axe = 0
            self.research_axe = self.axe_X

        # Back right
        elif angle >= 5 * np.pi / 8 and angle < 7 * np.pi / 8:
            logger.info(f"↘")
            pass

        # Back
        elif angle >= 7 * np.pi / 8 or angle < -7 * np.pi / 8:
            logger.info(f"⬇")
            if fctx.over_pad:
                self.target_pad = Vec2(
                    self.init_pos.x - self.pad_width, self.init_pos.y
                )
            else:
                self.target_pad = Vec2(
                    self.init_pos.x + self.pad_width, self.init_pos.y
                )
            self.platform_x_found = True
            self.change_axe = 0
            self.research_axe = self.axe_Y


class ReturnHome(State):
    def start(self, fctx: FlightContext):
        assert fctx.home_pad is not None

        # kalman is reset when the motors stop at the top pad RAJOUTER UNE FONCTION
        fctx.trajectory.position = fctx.home_pad

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            return GoLower()

        return None


class Stop(State):
    def next(self, _) -> State | None:
        return None
