from asyncio import Event
from dataclasses import dataclass, field

from .drone import Drone
from .types import Sensors
from .utils.observable import Broadcast


@dataclass
class Context:
    drone: Drone
    new_data: Event

    debug_tick: bool = False
    outlet: Broadcast = field(default_factory=Broadcast)
    sensors: Sensors = field(default_factory=Sensors)
