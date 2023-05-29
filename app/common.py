from asyncio import Event
from dataclasses import dataclass, field

from .drone import Drone
from .types import Sensors
from .utils.observable import Broadcast


@dataclass
class Context:
    """
    Shared context that is used to coordinate different modules without any
    inter-module dependencies. Provides access to the Drone class, latest
    sensor data and other minor shared resources.
    """

    drone: Drone
    new_data: Event

    debug_tick: bool = False
    outlet: Broadcast = field(default_factory=Broadcast)
    sensors: Sensors = field(default_factory=Sensors)
