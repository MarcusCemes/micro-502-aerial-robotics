from .context import Context


class FlightController:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    def update(self) -> None:
        pass

    def apply_flight_command(self) -> None:
        pass
