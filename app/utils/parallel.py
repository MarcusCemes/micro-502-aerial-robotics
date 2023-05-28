from multiprocessing import Process, Queue
from queue import Empty
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class LazyWorker(Generic[T, R]):
    def __init__(self, fn: Callable[[T], R]):
        super().__init__()

        self._tx: Queue = Queue(2)
        self._rx: Queue = Queue(1)
        self._fn = fn
        self._process = None

    def start(self):
        assert self._process is None

        self._process = Process(
            target=run,
            args=(self._fn, self._tx, self._rx),
        )

        self._process.start()

    def stop(self):
        assert self._process is not None

        self._process.terminate()

    def try_send(self, item: T) -> None:
        if self._tx.empty():
            self._tx.put_nowait(item)

    def collect(self) -> R | None:
        try:
            return self._rx.get_nowait()

        except Empty:
            return None


def run(fn: Callable[[T], R], rx: Queue, tx: Queue) -> None:
    while True:
        item: T = rx.get()
        result: R = fn(item)
        tx.put(result)
