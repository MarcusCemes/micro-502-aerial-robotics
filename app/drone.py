from __future__ import annotations

import threading
from asyncio import Event, Future, get_event_loop, get_running_loop, sleep
from dataclasses import replace
from enum import Enum
from typing import Any

from loguru import logger

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from .config import (
    CACHE_DIR,
    INITIAL_POSITION,
    RANGE_LOG_PERIOD_MS,
    STAB_LOG_PERIOD_MS,
    URI,
)
from .types import Sensors
from .utils.math import Vec2, deg_to_rad, mm_to_m


class LogNames(Enum):
    Stabiliser = "stab"
    Range = "range"


STAB_SENSORS = [
    ("stateEstimate.x", "float"),
    ("stateEstimate.y", "float"),
    ("stateEstimate.z", "float"),
    ("stabilizer.yaw", "float"),
    ("range.front", "uint16_t"),
    ("range.back", "uint16_t"),
    ("range.left", "uint16_t"),
    ("range.right", "uint16_t"),
]

RANGE_SENSORS = [
    ("range.zrange", "uint16_t"),
    ("stateEstimate.roll", "float"), 
    ("stateEstimate.pitch", "float"),
    ("stateEstimate.vx", "float"),
    ("stateEstimate.vy", "float")
]


class Drone:
    def __init__(self, data_event: Event) -> None:
        self.cf = Crazyflie(rw_cache=CACHE_DIR)

        self._connection_future: Future[Any] | None = None
        self._data_event = data_event
        self._last_sensor_data: Sensors = Sensors()
        self._lock = threading.Lock()
        self._loop = get_running_loop()

        self._sensors = Sensors()

        self.fast_speed = False
        self.first_landing = False
        self.last_z: float = 0

    async def __aenter__(self) -> Drone:
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        self.disconnect()

    async def connect(self) -> None:
        assert self._connection_future is None

        logger.info("📶 Connecting...")

        self._connection_future = get_event_loop().create_future()

        self.cf.connected.add_callback(self._on_connect)
        self.cf.disconnected.add_callback(self._on_disconnect)
        self.cf.connection_failed.add_callback(self._on_connection_failed)
        self.cf.connection_lost.add_callback(self._on_connection_lost)

        self.cf.open_link(URI)

        try:
            await self._connection_future
        finally:
            self._connection_future = None

    def configure_logging(self) -> None:
        assert self.cf.is_connected()

        for name, sensors, period in [
            (LogNames.Stabiliser, STAB_SENSORS, STAB_LOG_PERIOD_MS),
            (LogNames.Range, RANGE_SENSORS, RANGE_LOG_PERIOD_MS),
        ]:
            cfg = self._generate_logging_config(name, sensors, period)

            self.cf.log.add_config(cfg)

            cfg.data_received_cb.add_callback(self._on_sensor_data)
            cfg.error_cb.add_callback(self._on_sensor_error)

            cfg.start()

    def disconnect(self) -> None:
        if self.cf.is_connected():
            self.cf.close_link()
            logger.info("📶 Link closed")

    def get_last_sensor_reading(self) -> Sensors:
        with self._lock:
            return self._last_sensor_data

    async def reset_estimator(self, pos: Vec2 = Vec2(*INITIAL_POSITION)) -> None:
        logger.debug("🗿 Resetting Kalman estimator...")

        self.cf.param.set_value("kalman.initialX", f"{pos.x:.2f}")
        self.cf.param.set_value("kalman.initialY", f"{pos.y:.2f}")
        self.cf.param.set_value("kalman.resetEstimation", "1")
        await sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")

    # == Private == #

    def _generate_logging_config(
        self, name: LogNames, sensors: list[tuple[str, str]], period
    ) -> LogConfig:
        cfg = LogConfig(name=name, period_in_ms=period)

        for var, type in sensors:
            cfg.add_variable(var, type)

        return cfg

    # == Callbacks == #

    def _on_connect(self, _) -> None:
        logger.debug("Connected")

        if self._connection_future is not None:
            self._connection_future.get_loop().call_soon_threadsafe(
                self._connection_future.set_result, None
            )

    def _on_disconnect(self, _) -> None:
        pass

    def _on_connection_failed(self, _, e: str) -> None:
        logger.warning(f"Connection failed")

        if self._connection_future is not None:
            self._connection_future.get_loop().call_soon_threadsafe(
                self._connection_future.set_exception, ConnectionError(e)
            )

    def _on_connection_lost(self, _, msg: str) -> None:
        logger.warning(f"Connection lost: {msg}")

    def _on_sensor_data(self, _, data, cfg: LogConfig) -> None:
        match cfg.name:
            case LogNames.Stabiliser:
                self._sensors.x = data["stateEstimate.x"]
                self._sensors.y = data["stateEstimate.y"]
                self._sensors.z = data["stateEstimate.z"]
                self._sensors.back = float(mm_to_m(data["range.back"]))
                self._sensors.front = float(mm_to_m(data["range.front"]))
                self._sensors.left = float(mm_to_m(data["range.left"]))
                self._sensors.right = float(mm_to_m(data["range.right"]))
                self._sensors.yaw = deg_to_rad(data["stabilizer.yaw"])

            case LogNames.Range:
                self._sensors.down = float(mm_to_m(data["range.zrange"]))
                self._sensors.roll = float(deg_to_rad(data["stateEstimate.roll"]))
                self._sensors.pitch = float(deg_to_rad(data["stateEstimate.pitch"]))
                self._sensors.vx = float(data["stateEstimate.vx"])
                self._sensors.vy = float(data["stateEstimate.vy"])

        with self._lock:
            self._last_sensor_data = replace(self._sensors)

        self._loop.call_soon_threadsafe(self._data_event.set)

    def _on_sensor_error(self, cfg: LogConfig, msg: str):
        logger.warning(f"Sensor logging error ({cfg.name}): {msg}")
