from dataclasses import dataclass

Coords = tuple[int, int]


@dataclass
class Sensors:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    front: int = 0
    back: int = 0
    left: int = 0
    right: int = 0
    down: int = 0
