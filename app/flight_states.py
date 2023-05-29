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
    INITIAL_POSITION,
    POSITION_ERROR,
    LATERAL_MOVEMENT,
    POSITION_ERROR_PAD,
    PAD_WIDTH,
    LINE_TARGET_SEARCH,
    HOME_PAD_ERROR,
    POSITION_ERROR_PAD,
    MIN_DIFF,
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


def pad_detection(fctx: FlightContext):
    # logger.debug(f"last z is {fctx.ctx.drone.last_z}, z position: {fctx.ctx.sensors.z}")
    # correction = 0  # = np.cos(fctx.ctx.sensors.pitch) * np.cos(fctx.ctx.sensors.roll)
    logger.warning(f"Dif {np.sum(np.abs(np.diff(fctx.ctx.drone.down_hist)))}")
    if np.sum(np.abs(np.diff(fctx.ctx.drone.down_hist))) > MIN_DIFF:
        logger.info(f"🎯 Detected pad!")
        # fctx.ctx.drone.last_z = correction * fctx.ctx.sensors.down
        return True
    # fctx.ctx.drone.last_z = correction * fctx.ctx.sensors.down

    return False


# == Types == #


@dataclass
class Trajectory:
    altitude: float = 0.0
    orientation: float = 0.0
    position: Vec2 = field(default_factory=Vec2)
    touch_down: bool = False


@dataclass
class FlightContext:
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
    # z_hist = np.zeros(5)
    enable_path_finding: bool = True

    # == Sensors == #

    def is_near_next_waypoint(self, error=POSITION_ERROR) -> bool:
        if self.path is None or len(self.path) == 0:
            return False
        else:
            return self.is_near_position(self.path[0], error)

    def is_near_target(self, error=POSITION_ERROR) -> bool:
        # logger.debug(f"True position: {self.get_position()}")
        # logger.debug(f"Target position: {self.trajectory.position}")
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

    def plot_hist(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        ax1.plot(self.ctx.drone.tot_down_hist)
        ax1.set_title("total down values")
        ax2.plot(self.ctx.drone.tot_mean)
        ax2.set_title("total mean")
        ax3.plot(self.ctx.drone.tot_diff)
        ax3.set_title("total diff")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"app/output/plot_{self.id+1}.png")
        with open(f"app/output/plot_{self.id}.json", "w+") as p:
            json.dump(self.ctx.drone.tot_diff, p)
        self.ctx.drone.tot_down_hist = []
        self.ctx.drone.tot_mean = []
        self.ctx.drone.tot_diff = []
        self.id += 1

    def detect_on_hist():
        pass


class State(Protocol):
    def start(self, fctx: FlightContext) -> None:
        return

    def next(self, fctx: FlightContext) -> State | None:
        ...


# == States == #


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


class Boot(State):
    def next(self, fctx: FlightContext) -> State | None:
        (x, y) = INITIAL_POSITION
        fctx.trajectory.position = Vec2(x, y)
        fctx.home_pad = fctx.get_position()

        # logger.warning(f"Voilà {np.sum(np.diff(fctx.ctx.sensors.down_hist))}")
        # logger.warning(f"Voilà voilà {fctx.ctx.sensors.down_hist}")
        return Takeoff()


class Takeoff(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.altitude = 0.4

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_altitude():
            fctx.pad_detection = True
            return Cross()
        return None


class MoveForward(State):
    def __init__(self) -> None:
        pass

    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.altitude = 0.35
        fctx.trajectory.position = fctx.navigation.global_position() + Vec2(2, 0)

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            fctx.plot_hist()
            return MoveBackward()


class MoveBackward(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.position = fctx.navigation.global_position() + Vec2(-2, 0)

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target():
            return MoveForward()


class Cross(State):
    def start(self, fctx: FlightContext) -> None:
        logger.info("🛫🏢🏢 Crossing")
        fctx.trajectory.position = Vec2(4.5, 1.5)
        fctx.scan = True

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.has_crossed_the_line():
            return TargetSearch()


class TargetSearch(State):
    def __init__(self):
        self.research_points = []
        self.map_offset = 3.5
        self.index = 0

    def start(self, fctx: FlightContext):
        fctx.scan = False
        # compute target map
        self.compute_target_map(fctx)
        # set target
        fctx.trajectory.position = fctx.navigation.to_position(
            self.research_points[self.index]
        )

    def next(self, fctx: FlightContext):
        self.update_research_point(fctx)

        # logger.debug(f"Pos {fctx.navigation.global_position()}")

        fctx.ctx.drone.prob_map.fill(fctx)

        # if np.amax(fctx.ctx.drone.prob_map.probability_map) > PROBABILITY_THRESHOLD:
        #     fctx.target_pad = fctx.ctx.drone.prob_map.find_mean_position()
        #     return Centering()

        if fctx.is_near_target():
            self.index = self.index + 1

            # logger.debug(f"max prob {np.max(fctx.ctx.drone.prob_map.probability_map)}")
            fctx.ctx.drone.prob_map.save()

            if self.index == len(self.research_points):
                logger.info("🔎 look at that")
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

        research_points2 = [
            # (4.0, 0.8),
            # (4.0, 1.5),
            # (4.0, 2.3),
            # (4.4, 2.3),
            # (4.4, 1.5),
            # (4.4, 0.8),
        ]

        research_points3 = [
            # (4.7, 0.8),
            # (4.7, 1.5),
            # (4.7, 2.3),
            # (4.4, 2.7),
            # (4.4, 1.9),
            # (4.4, 1.1),
            # (4.4, 0.3),
            # (4.2, 0.8),
            # (4.2, 1.5),
            # (4.2, 2.3),
            # (4.0, 2.7),
            # (4.0, 1.9),
            # (4.0, 1.1),
            # (4.0, 0.3),
            # (3.8, 0.8),
            # (3.8, 1.5),
            # (3.8, 2.3),
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
        # print(research_points)
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


class Centering(State):
    def __init__(self):
        self.UP: Final = 0
        self.DOWN: Final = 1

        self.X: Final = 0
        self.Y: Final = 1

    def start(self, fctx: FlightContext):
        # logger.info("🏴Centering")
        self.target_pad = fctx.target_pad

        self.research_axe: int = self.X
        self.research_dir: int = self.UP
        self.research_counter: int = 0
        self.counter_axe: int = 0

        fctx.enable_path_finding = False
        fctx.scan = False

    def next(self, fctx: FlightContext) -> State | None:
        if self.centering(fctx):
            fctx.target_pad = fctx.ctx.drone.prob_map.find_mean_position()
            logger.debug(f"Pad found at {fctx.target_pad}")
            return GoToTarget()
        return None

    def set_target(self, fctx: FlightContext):
        if self.research_axe == self.X:
            if self.research_dir == self.UP:
                vect = Vec2(LATERAL_MOVEMENT, 0)
            else:
                vect = Vec2(-LATERAL_MOVEMENT, 0)

        else:
            if self.research_dir == self.UP:
                vect = Vec2(0, LATERAL_MOVEMENT)
            else:
                vect = Vec2(0, -LATERAL_MOVEMENT)

        fctx.trajectory.position = self.target_pad + vect

    def centering(self, fctx: FlightContext):
        fctx.ctx.drone.prob_map.fill(fctx)

        # Update axis
        if self.counter_axe >= 2:
            if self.research_axe == self.X:
                self.research_axe = self.Y
            else:
                self.research_axe = self.X

            self.counter_axe = 0
            self.research_dir = self.UP
            self.research_counter += 1

        if self.research_counter >= 2:
            fctx.ctx.drone.prob_map.save()
            return True

        # Update target point
        self.set_target(fctx)

        # Update direction
        if self.research_axe == self.X:
            if self.research_dir == self.UP:
                if fctx.is_near_target_pad():
                    self.counter_axe += 1
                    self.research_dir = self.DOWN

            else:
                if fctx.is_near_target_pad():
                    self.research_dir = self.UP

        elif self.research_axe == self.Y:
            if self.research_dir == self.UP:
                if fctx.is_near_target_pad():
                    self.counter_axe += 1
                    self.research_dir = self.DOWN

            else:
                if fctx.is_near_target_pad():
                    self.research_dir = self.UP

        return False


class GoToTarget(State):
    def start(self, fctx: FlightContext) -> None:
        fctx.trajectory.position = fctx.target_pad

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_pad():
            return TouchDown()


class TargetCentering(State):
    def __init__(self):
        self.UP: Final = 0
        self.DOWN: Final = 1

        self.X: Final = 0
        self.Y: Final = 1

    def start(self, fctx: FlightContext):
        self.target_pad = fctx.target_pad

        self.platform_x_found: bool = False
        self.platform_y_found: bool = False
        self.detection: bool = False

        self.research_axe: int = self.X
        self.research_dir: int = self.UP
        self.research_counter: int = 0
        self.counter_axe: int = 0

        self.update_platform_pos(fctx)
        # fctx.ctx.drone.slow_speed = True

        fctx.enable_path_finding = False
        fctx.scan = False

    def next(self, fctx: FlightContext) -> State | None:
        if self.platform_x_found and self.platform_y_found:
            fctx.target_pad = (
                self.target_pad
            )  # on enregistre la position du pad pour le retour
            fctx.trajectory.position = self.target_pad
            if fctx.is_near_target_pad():
                logger.info(f"🐣Target pad found {self.target_pad}")
                return TouchDown()
        else:
            return self.centering(fctx)

    def set_target(self, fctx: FlightContext):
        if self.research_axe == self.X:
            if self.research_dir == self.UP:
                vect = Vec2(LATERAL_MOVEMENT, 0)
            else:
                vect = Vec2(-LATERAL_MOVEMENT, 0)

        else:
            if self.research_dir == self.UP:
                vect = Vec2(0, LATERAL_MOVEMENT)
            else:
                vect = Vec2(0, -LATERAL_MOVEMENT)

        fctx.trajectory.position = self.target_pad + vect

    def centering(self, fctx: FlightContext):
        """
        Find the center of the platform.

        Args:
            sensor_data (Dictionarie): data sensors of the drone

        Returns:
            commande (List): control commande of the drone
            (Boolean): Center found or not
        """

        # Update pad position
        if pad_detection(fctx) and self.detection:
            self.update_platform_pos(fctx)
            # fctx.ctx.drone.fast_speed = False
            self.detection = False

        # Update axis
        if self.counter_axe >= 2:
            if self.research_axe == self.X:
                self.research_axe = self.Y
            else:
                self.research_axe = self.X

            self.counter_axe = 0
            self.research_dir = self.UP
            self.research_counter += 1
            # fctx.ctx.drone.fast_speed = False
            self.detection = False

        # Check if stuck
        if self.research_counter >= 3:
            logger.info(f"🔒 Stuck !!!")
            return TargetSearch()

        # if fctx.is_near_position(fctx.target_pad, POSITION_ERROR_PAD):
        #     self.detection = False

        # if self.detection:
        #     logger.debug("🛸Looking for the edge")

        # Update target point
        self.set_target(fctx)

        # Update direction
        if self.research_axe == self.X:
            if self.research_dir == self.UP:
                if fctx.is_near_target_pad():
                    self.counter_axe += 1
                    self.research_dir = self.DOWN
                    # fctx.ctx.drone.fast_speed = False
                    self.detection = False

            else:
                if fctx.is_near_target_pad():
                    self.research_dir = self.UP
                    # fctx.ctx.drone.fast_speed = True
                    self.detection = True

        elif self.research_axe == self.Y:
            if self.research_dir == self.UP:
                if fctx.is_near_target_pad():
                    self.counter_axe += 1
                    self.research_dir = self.DOWN
                    # fctx.ctx.drone.fast_speed = False
                    self.detection = False

            else:
                if fctx.is_near_target_pad():
                    self.research_dir = self.UP
                    # fctx.ctx.drone.fast_speed = True
                    self.detection = True

    def update_platform_pos(self, fctx: FlightContext):
        """
        Update the position of the platform from the actual position
        and the direction of the movement.

        Args:
            position (List): actual position at the moment of the call
        """

        angle = -math.atan2(fctx.ctx.sensors.vy, fctx.ctx.sensors.vx)
        # logger.debug(f"angle {angle}")

        # Back left
        if angle >= -7 * np.pi / 8 and angle < -5 * np.pi / 8:
            logger.info(f"↙")
            pass

        # Left
        elif angle >= -5 * np.pi / 8 and angle < -3 * np.pi / 8:
            logger.info(f"⬅")
            self.target_pad.y = fctx.navigation.global_position().y + PAD_WIDTH / 2
            self.platform_y_found = True
            self.counter_axe = 0
            self.research_axe = self.X

        # Front left
        elif angle >= -3 * np.pi / 8 and angle < -np.pi / 8:
            logger.info(f"↖")
            pass

        # Front
        elif angle >= -np.pi / 8 and angle < np.pi / 8:
            logger.info(f"⬆")
            self.target_pad.x = fctx.navigation.global_position().x + PAD_WIDTH / 2
            self.platform_x_found = True
            self.counter_axe = 0
            self.research_axe = self.Y

        # Front right
        elif angle >= np.pi / 8 and angle < 3 * np.pi / 8:
            logger.info(f"↗")
            pass

        # Right
        elif angle >= 3 * np.pi / 8 and angle < 5 * np.pi / 8:
            logger.info(f"➡")
            self.target_pad.y = fctx.navigation.global_position().y - PAD_WIDTH / 2
            self.platform_y_found = True
            self.counter_axe = 0
            self.research_axe = self.X

        # Back right
        elif angle >= 5 * np.pi / 8 and angle < 7 * np.pi / 8:
            logger.info(f"↘")
            pass

        # Back
        elif angle >= 7 * np.pi / 8 or angle < -7 * np.pi / 8:
            logger.info(f"⬇")
            self.target_pad.x = fctx.navigation.global_position().x - PAD_WIDTH / 2
            self.platform_x_found = True
            self.counter_axe = 0
            self.research_axe = self.Y


class TouchDown(State):
    def start(self, fctx: FlightContext):
        logger.info(f"👌 Touching down")
        self.touched = False
        fctx.trajectory.position = fctx.target_pad

    def next(self, fctx: FlightContext) -> State | None:
        fctx.trajectory.altitude -= 0.005
        if fctx.trajectory.altitude < 0.001 and not self.touched:
            fctx.trajectory.altitude = 0.4
            self.touched = True

        if fctx.is_near_target_altitude() and self.touched:
            return ReturnHome()

        return None


class ReturnHome(State):
    def start(self, fctx: FlightContext):
        fctx.scan = True
        fctx.enable_path_finding = True
        fctx.ctx.drone.slow_speed = False
        assert fctx.home_pad is not None
        logger.info(f"🏠 Returning home to {fctx.home_pad}")
        fctx.trajectory.position = fctx.home_pad
        # print(f"home pad:  {fctx.home_pad.x}, {fctx.home_pad.y}")
        logger.info(f"drone position: {fctx.navigation.global_position().x}, {fctx.navigation.global_position().y}")

    def next(self, fctx: FlightContext) -> State | None:
        if (
            fctx.is_near_target_pad()
        ):  # or (pad_detection(fctx) and fctx.is_near_home()):
            logger.info(f"drone position: {fctx.navigation.global_position().x}, {fctx.navigation.global_position().y}")
            fctx.target_pad = fctx.home_pad
            return GoLower()

        return None


class HomeSearch(State):
    def __init__(self):
        self.research_points = []
        self.index = 0

    def start(self, fctx: FlightContext):
        fctx.scan = True
        # compute target map
        self.compute_target_map(fctx)
        # set target
        fctx.trajectory.position = fctx.navigation.to_position(
            self.research_points[self.index]
        )

    def next(self, fctx: FlightContext):
        self.update_research_point(fctx)
        if self.index == len(self.research_points):
            # logger.info("No target found")
            return Stop()
        if fctx.is_near_target_pad():
            # move to next target point
            # print("no home found on this point")
            self.index = self.index + 1
        fctx.trajectory.position = fctx.navigation.to_position(
            self.research_points[self.index]
        )

        if pad_detection(fctx):
            fctx.target_pad = fctx.navigation.global_position()
            logger.debug(f"🤣First detection {fctx.target_pad}")
            return GoLower(fctx)

    def compute_target_map(self, fctx: FlightContext):
        research_points_temp = [
            (fctx.home_pad.x + 0.5, fctx.home_pad.y),
            (fctx.home_pad.x + 0.25, fctx.home_pad.y),
            (fctx.home_pad.x, fctx.home_pad.y),
            (fctx.home_pad.x + -0.25, fctx.home_pad.y),
            (fctx.home_pad.x + -0.5, fctx.home_pad.y),
            (fctx.home_pad.x + -0.5, fctx.home_pad.y - 0.25),
            (fctx.home_pad.x + -0.25, fctx.home_pad.y - 0.25),
            (fctx.home_pad.x + -0, fctx.home_pad.y - 0.25),
            (fctx.home_pad.x + 0.25, fctx.home_pad.y - 0.25),
            (fctx.home_pad.x + 0.5, fctx.home_pad.y - 0.25),
            (fctx.home_pad.x + 0.5, fctx.home_pad.y + 0.25),
            (fctx.home_pad.x + 0.25, fctx.home_pad.y + 0.25),
            (fctx.home_pad.x + 0.0, fctx.home_pad.y + 0.25),
            (fctx.home_pad.x - 0.25, fctx.home_pad.y + 0.25),
            (fctx.home_pad.x - 0.5, fctx.home_pad.y + 0.25),
        ]

        research_points = []

        for i in range(len(research_points_temp)):
            point = fctx.navigation.to_coords(
                Vec2(research_points_temp[i][0], research_points_temp[i][1])
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


class HomeCentering(State):
    def __init__(self, fctx: FlightContext):
        self.UP: Final = 0
        self.DOWN: Final = 1

        self.X: Final = 0
        self.Y: Final = 1

    def start(self, fctx: FlightContext):
        self.target_pad = fctx.target_pad

        self.platform_x_found: bool = False
        self.platform_y_found: bool = False
        self.detection: bool = False

        self.research_axe: int = self.X
        self.research_dir: int = self.UP
        self.research_counter: int = 0
        self.counter_axe: int = 0

        self.update_platform_pos(fctx)
        fctx.ctx.drone.slow_speed = True
        logger.debug("🚧 Slow speed mode")

        fctx.scan = False

    def next(self, fctx: FlightContext) -> State | None:
        if self.platform_x_found and self.platform_y_found:
            fctx.target_pad = (
                self.target_pad
            )  # on enregistre la position du pad pour le retour
            fctx.trajectory.position = self.target_pad
            if fctx.is_near_target_pad():
                logger.info(f"🐣Target pad found {self.target_pad}")
                return GoLower()
        else:
            return self.centering(fctx)

    def set_target(self, fctx: FlightContext):
        if self.research_axe == self.X:
            if self.research_dir == self.UP:
                vect = Vec2(LATERAL_MOVEMENT, 0)
            else:
                vect = Vec2(-LATERAL_MOVEMENT, 0)

        else:
            if self.research_dir == self.UP:
                vect = Vec2(0, LATERAL_MOVEMENT)
            else:
                vect = Vec2(0, -LATERAL_MOVEMENT)

        fctx.trajectory.position = self.target_pad + vect

    def centering(self, fctx: FlightContext):
        """
        Find the center of the platform.

        Args:
            sensor_data (Dictionarie): data sensors of the drone

        Returns:
            commande (List): control commande of the drone
            (Boolean): Center found or not
        """

        # Update pad position
        if pad_detection(fctx) and self.detection:
            self.update_platform_pos(fctx)
            self.detection = False

        # Update axis
        if self.counter_axe >= 2:
            if self.research_axe == self.X:
                self.research_axe = self.Y
            else:
                self.research_axe = self.X

            self.counter_axe = 0
            self.research_dir = self.UP
            self.research_counter += 1
            self.detection = False

        # Check if stuck
        if self.research_counter >= 3:
            logger.info(f"🔒 Stuck !!!")
            # return TargetSearch()

        if fctx.is_near_position(fctx.target_pad, POSITION_ERROR_PAD):
            self.detection = False

        # Update target point
        self.set_target(fctx)

        # Update direction
        if self.research_axe == self.X:
            if self.research_dir == self.UP:
                if fctx.is_near_target_pad():
                    self.counter_axe += 1
                    self.research_dir = self.DOWN
                    self.detection = True

            else:
                if fctx.is_near_target_pad():
                    self.research_dir = self.UP
                    self.detection = True

        elif self.research_axe == self.Y:
            if self.research_dir == self.UP:
                if fctx.is_near_target_pad():
                    self.counter_axe += 1
                    self.research_dir = self.DOWN
                    self.detection = True

            else:
                if fctx.is_near_target_pad():
                    self.research_dir = self.UP
                    self.detection = True

    def update_platform_pos(self, fctx: FlightContext):
        """
        Update the position of the platform from the actual position
        and the direction of the movement.

        Args:
            position (List): actual position at the moment of the call
        """

        angle = -math.atan2(fctx.ctx.sensors.vy, fctx.ctx.sensors.vx)
        logger.debug(f"angle {angle}")

        # Back left
        if angle >= -7 * np.pi / 8 and angle < -5 * np.pi / 8:
            logger.info(f"↙")
            pass

        # Left
        elif angle >= -5 * np.pi / 8 and angle < -3 * np.pi / 8:
            logger.info(f"⬅")
            self.target_pad.y = fctx.navigation.global_position().y + PAD_WIDTH / 2
            self.platform_y_found = True
            self.counter_axe = 0
            self.research_axe = self.X

        # Front left
        elif angle >= -3 * np.pi / 8 and angle < -np.pi / 8:
            logger.info(f"↖")
            pass

        # Front
        elif angle >= -np.pi / 8 and angle < np.pi / 8:
            logger.info(f"⬆")
            self.target_pad.x = fctx.navigation.global_position().x + PAD_WIDTH / 2
            self.platform_x_found = True
            self.counter_axe = 0
            self.research_axe = self.Y

        # Front right
        elif angle >= np.pi / 8 and angle < 3 * np.pi / 8:
            logger.info(f"↗")
            pass

        # Right
        elif angle >= 3 * np.pi / 8 and angle < 5 * np.pi / 8:
            logger.info(f"➡")
            self.target_pad.y = fctx.navigation.global_position().y - PAD_WIDTH / 2
            self.platform_y_found = True
            self.counter_axe = 0
            self.research_axe = self.X

        # Back right
        elif angle >= 5 * np.pi / 8 and angle < 7 * np.pi / 8:
            logger.info(f"↘")
            pass

        # Back
        elif angle >= 7 * np.pi / 8 or angle < -7 * np.pi / 8:
            logger.info(f"⬇")
            self.target_pad.x = fctx.navigation.global_position().x - PAD_WIDTH / 2
            self.platform_x_found = True
            self.counter_axe = 0
            self.research_axe = self.Y


# class GoForward(State):
#     def start(self, fctx: FlightContext) -> None:
#         fctx.trajectory.position.x = 2.0
#         # fctx.trajectory.orientation = pi

#         fctx.scan = True

#     def next(self, fctx: FlightContext) -> State | None:
#         if fctx.is_near_target():
#             return ReturnHome()

#         return None


# class Pause(State):
#     def start(self, fctx: FlightContext) -> None:
#         pass

#         fctx.scan = False

#     def next(self, fctx: FlightContext) -> State | None:
#         fctx.trajectory.position = fctx.target_pad
#         return None


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


class TargetCenteringEasy(State):
    def __init__(self):
        pass

    def start(self, fctx: FlightContext):
        self.pad_pos = fctx.navigation.global_position() + Vec2(
            fctx.ctx.sensors.vx, fctx.ctx.sensors.vy
        ).set_mag(0.15)
        fctx.trajectory.position = self.pad_pos

    def next(self, fctx: FlightContext) -> State | None:
        if fctx.is_near_target_pad():
            return TouchDown()
        return None
