from dataclasses import dataclass, field
from typing import Any, Callable

QUEUE_SIZE = 32


Subscriber = Callable[[Any], None]
Unsubscriber = Callable[[], None]


class Observable:
    def __init__(self):
        self._observers: set[Subscriber] = set()

    def subscribe(self, fn: Subscriber):
        self._observers.add(fn)

    def unregister(self, fn: Subscriber):
        self._observers.remove(fn)


class Broadcast(Observable):
    def __init__(self):
        super().__init__()

    def broadcast(self, payload: Any):
        for fn in self._observers:
            try:
                fn(payload)
            except:
                pass


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
    ticks: int = 0
