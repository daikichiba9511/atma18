import numpy as np
import numpy.typing as npt


def score(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
) -> float:
    """mean abs erros

    Args:
        y_true (npt.NDArray): true value. (x0, y0, z0, x1, y1, z1, ..., x5, y5, z5)
        y_pred (npt.NDArray): predicted value. (x0, y0, z0, x1, y1, z1, ..., x5, y5, z5)
    """
    return np.mean(np.abs(y_true - y_pred).reshape(-1)).item()
