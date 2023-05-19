from dataclasses import dataclass

Coords = tuple[int, int]


@dataclass
class Sensors:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    front: float = 0
    back: float = 0
    left: float = 0
    right: float = 0
    down: float = 0
