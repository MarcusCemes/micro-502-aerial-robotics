from __future__ import annotations

import threading
from asyncio import (
    AbstractEventLoop,
    Event,
    Future,
    get_event_loop,
    get_running_loop,
    sleep,
)
from enum import Enum
from typing import Any

from loguru import logger

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from .config import CACHE_DIR, INITIAL_POSITION, LOG_PERIOD_MS, URI
from .types import Sensors
from .utils.math import deg_to_rad, mm_to_m


class LogNames(Enum):
    Stabiliser = "stab"


SENSORS = [
    ("stateEstimate.x", "float"),
    ("stateEstimate.y", "float"),
    ("stateEstimate.z", "float"),
    ("stabilizer.yaw", "float"),
    # ("stabilizer.pitch", "float"),
    # ("stabilizer.roll", "float"),
    ("range.front", "uint16_t"),
    ("range.back", "uint16_t"),
    ("range.left", "uint16_t"),
    ("range.right", "uint16_t"),
    ("range.zrange", "uint16_t"),
]


class Drone:
    def __init__(self, data_event: Event) -> None:
        self.cf = Crazyflie(rw_cache=CACHE_DIR)

        self._connection_future: Future[Any] | None = None
        self._data_event = data_event
        self._last_sensor_data: Sensors | None = None
        self._lock = threading.Lock()
        self._loop = get_running_loop()
        self.first_landing = False 
        self.slow_speed = False

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

        cfg = self._generate_logging_config()

        self.cf.log.add_config(cfg)

        cfg.data_received_cb.add_callback(self._on_sensor_data)
        cfg.error_cb.add_callback(self._on_sensor_error)

        cfg.start()

    def disconnect(self) -> None:
        if self.cf.is_connected():
            self.cf.close_link()
            logger.info("📶 Link closed")

    def get_last_sensor_reading(self) -> Sensors | None:
        with self._lock:
            return self._last_sensor_data

    async def reset_estimator(self) -> None:
        logger.debug("🗿Reseting Kalman estimator...")

        (x, y) = INITIAL_POSITION
        self.cf.param.set_value("kalman.initialX", f"{x:.2f}")
        self.cf.param.set_value("kalman.initialY", f"{y:.2f}")
        self.cf.param.set_value("kalman.resetEstimation", "1")
        await sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")
        
    async def comeback(self, start_x, start_y):
        logger.debug("⚙️Resetting position for second take off.")
        
        self.cf.param.set_value("kalman.initialX", f"{start_x:.2f}")
        self.cf.param.set_value("kalman.initialY", f"{start_y:.2f}")
        self.cf.param.set_value("kalman.resetEstimation", "1") ##
        await sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")

    # == Private == #

    def _generate_logging_config(self) -> LogConfig:
        cfg = LogConfig(name=LogNames.Stabiliser, period_in_ms=LOG_PERIOD_MS)

        for name, type in SENSORS:
            cfg.add_variable(name, type)

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
                if self._last_sensor_data is not None:
                    previous_x = self._last_sensor_data.x
                    previous_y = self._last_sensor_data.y
                else:
                    previous_x : float = 0
                    previous_y : float = 0
                reading = Sensors(                    
                    x=data["stateEstimate.x"],
                    y=data["stateEstimate.y"],
                    z=data["stateEstimate.z"],
                    vx=data["stateEstimate.x"]-previous_x,
                    vy=data["stateEstimate.y"]-previous_y,
                    back=float(mm_to_m(data["range.back"])),
                    front=float(mm_to_m(data["range.front"])),
                    left=float(mm_to_m(data["range.left"])),
                    right=float(mm_to_m(data["range.right"])),
                    down=float(mm_to_m(data["range.zrange"])),
                    # pitch=deg_to_rad(data["stabilizer.pitch"]),
                    # roll=deg_to_rad(data["stabilizer.roll"]),
                    yaw=deg_to_rad(data["stabilizer.yaw"]),
                )

                with self._lock:
                    self._last_sensor_data = reading

                self._loop.call_soon_threadsafe(self._data_event.set)

    def _on_sensor_error(self, cfg: LogConfig, msg: str):
        logger.warning(f"Sensor logging error ({cfg.name}): {msg}")
