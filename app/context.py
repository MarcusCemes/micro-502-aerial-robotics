from asyncio import AbstractEventLoop, get_running_loop
from dataclasses import dataclass, field

from .drone import Drone
from .types import Sensors


@dataclass
class Context:
    drone: Drone = field(default_factory=Drone)
    loop: AbstractEventLoop = field(default_factory=get_running_loop)
    sensors: Sensors = field(default_factory=Sensors)
