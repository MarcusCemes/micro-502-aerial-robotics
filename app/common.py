from asyncio import Event
from dataclasses import dataclass, field

from .drone import Drone
from .types import Sensors


@dataclass
class Context:
    drone: Drone
    new_data: Event

    debug_tick: bool = False
    sensors: Sensors = field(default_factory=Sensors)
