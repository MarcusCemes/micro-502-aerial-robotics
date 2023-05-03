from dataclasses import dataclass, field

from utils import Broadcast

QUEUE_SIZE = 32


@dataclass
class Sensors:
    t: float = 0.0
    x_global: float = 0.0
    y_global: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    v_forward: float = 0.0
    v_left: float = 0.0
    range_front: float = 0.0
    range_left: float = 0.0
    range_back: float = 0.0
    range_right: float = 0.0
    range_down: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class Context:
    debug_tick: bool = False
    outlet: Broadcast = field(default_factory=Broadcast)
    sensors: Sensors = field(default_factory=Sensors)
