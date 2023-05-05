import numpy as np
import numpy.typing as npt
from config import DEBUG_FILES


def export_array(name: str, data: npt.NDArray, cmap: str = "gray", flip: bool = True):
    if not DEBUG_FILES:
        return

    try:
        from matplotlib.image import imsave

        if flip:
            data = np.flip(np.flip(data, 1), 0)

        imsave(f"output/{name}.png", data, cmap=cmap)

    except ImportError:
        pass
