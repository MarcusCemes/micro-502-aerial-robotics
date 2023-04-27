from typing import Any, Callable, Set

Subscriber = Callable[[Any], None]
Unsubscriber = Callable[[], None]


class Observable:
    def __init__(self):
        self._observers: Set[Subscriber] = set()

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
