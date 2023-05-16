# == Crazyflie configuration == #

from typing import Final

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
ANGULAR_LIMIT_DEG: Final = 90
