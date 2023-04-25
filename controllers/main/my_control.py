from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import numpy as np

AIRBORN_THRESHOLD = 0.48
TARGET_HEIGHT = 0.5


class SensorData(TypedDict):
    t: float
    x_global: float
    y_global: float
    roll: float
    pitch: float
    yaw: float
    v_forward: float
    v_left: float
    range_front: float
    range_left: float
    range_back: float
    range_right: float
    range_down: float
    yaw_rate: float


@dataclass
class ControlCommand:
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    yaw_rate: float = 0.0
    altitude: float = 0.0

    def to_list(self):
        return [self.velocity_x, self.velocity_y, self.yaw_rate, self.altitude]


class State(Enum):
    Boot = 0
    TakeOff = 1
    Hover = 2
    Land = 3


class MyController:
    def __init__(self):
        self.state = State.Boot

    def step_control(self, data: SensorData):
        self.data = data
        return self.next().to_list()

    def next(self) -> ControlCommand:
        match self.state:
            case State.Boot:
                print("Custom controller active!")
                self.update_state(State.TakeOff)
                return self.next()

            case State.TakeOff:
                if self.is_airborn():
                    self.update_state(State.Hover)
                    return self.next()

                return ControlCommand(altitude=TARGET_HEIGHT)

            case State.Hover:
                self.update_state(State.Land)
                return self.next()

            case State.Land:
                return ControlCommand(altitude=0)

    def is_airborn(self) -> bool:
        return self.data["range_down"] >= AIRBORN_THRESHOLD

    def update_state(self, state: State):
        print(f"[STATE] {state}")
        self.state = state
