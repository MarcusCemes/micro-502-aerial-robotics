from typing import Final

# == Crazyflie configuration == #

CACHE_DIR: Final = "./cache"
LOG_PERIOD_MS: Final = 50
URI: Final = "radio://0/10/2M/E7E7E7E711"


# == Environment == #

BOX_LIMIT: Final = 0.5


# == Flight == #

CRUISING_ALTITUDE: Final = 0.5

ALTITUDE_ERROR: Final = 0.05
POSITION_ERROR: Final = 0.05

PAD_THRESHOLD: Final = 0.1

VELOCITY_LIMIT: Final = 0.5
VERTICAL_VELOCITY_LIMIT: Final = 0.2
ANGULAR_VELOCITY_LIMIT_DEG: Final = 90.0
ANGULAR_SCAN_VELOCITY: Final = 45.0


# == Navigation == #

MAP_SIZE: Final = (5.0, 5.0)
OPTIMISE_PATH: Final = False
RANGE_THRESHOLD: Final = 2.0
MAP_PX_PER_M: Final = 25

# == Miscellaneous == #

DEBUG_FILES: Final = True
