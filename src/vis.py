import pathlib
from typing import Any, Final, Iterable, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib import axes

from src import utils

_T_NBit = TypeVar("_T_NBit", bound=npt.NBitBase)


def bar(
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int32],
    x_label: str,
    y_label: str,
    title: str,
    ax: axes.Axes | None = None,
    label_type: Literal["center", "edge"] = "edge",
    fig_size: tuple[int, int] = (15, 10),
    save_fp: pathlib.Path | None = None,
) -> None:
    """plot bar"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = None
    if ax is None:
        raise ValueError("ax is None")
    bars = ax.bar(x, y, alpha=0.7, label=y_label)
    ax.bar_label(bars, label_type=label_type)
    ax.set_title(title)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(list(map(str, x)))
    ax.legend()
    if fig is not None:
        fig.tight_layout()
        fig.show()
    if save_fp is not None and fig is not None:
        fig.savefig(save_fp)


def _complement_lack_classes(
    y: npt.NDArray[np.int32], y_cnt: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    for i in range(4):
        if i not in y:
            y = np.append(y, i)
            y_cnt = np.append(y_cnt, 0)
    idx = np.argsort(y)
    y = y[idx]
    y_cnt = y_cnt[idx]
    return y, y_cnt


def dists(
    y_true: npt.NDArray[np.int32],
    y_pred: npt.NDArray[np.int32],
    save_fp: pathlib.Path,
    figsize: tuple[int, int] = (30, 20),
) -> None:
    # -- 予測値 & 正解値の分布
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    y_true_values, y_true_counts = np.unique(y_true, return_counts=True)
    y_true_values, y_true_counts = _complement_lack_classes(y_true_values, y_true_counts)  # type: ignore
    bar(x=y_true_values, y=y_true_counts, x_label="target", y_label="count", title="y_true", ax=ax[0, 0])
    y_pred_values, y_pred_counts = np.unique(y_pred, return_counts=True)
    y_pred_values, y_pred_counts = _complement_lack_classes(y_pred_values, y_pred_counts)  # type: ignore
    bar(x=y_pred_values, y=y_pred_counts, x_label="target", y_label="count", title="y_pred", ax=ax[0, 1])

    ax02 = ax[0, 2]
    bars = ax02.bar(y_true_values, y_true_counts, alpha=0.5, label="y_true")
    ax02.bar_label(bars, label_type="center")
    bars = ax02.bar(y_pred_values, y_pred_counts, alpha=0.5, label="y_pred")
    ax02.bar_label(bars)
    ax02.set_title("y_true vs y_pred")
    ax02.set_xticks(y_true_values)
    ax02.set(xlabel="target", ylabel="count", xticklabels=list(map(str, y_true_values)))
    ax02.legend()

    # -- 予測値 & 正解値のうち間違えた部分の分布
    miss = y_true != y_pred
    miss_y_true = y_true[miss]
    miss_y_pred = y_pred[miss]

    miss_y_true_values, miss_y_true_counts = np.unique(miss_y_true, return_counts=True)
    miss_y_true_values, miss_y_true_counts = _complement_lack_classes(miss_y_true_values, miss_y_true_counts)
    bar(
        x=miss_y_true_values, y=miss_y_true_counts, x_label="target", y_label="count", title="miss_y_true", ax=ax[1, 0]
    )
    miss_y_pred_values, miss_y_pred_counts = np.unique(miss_y_pred, return_counts=True)
    miss_y_pred_values, miss_y_pred_counts = _complement_lack_classes(miss_y_pred_values, miss_y_pred_counts)
    bar(
        x=miss_y_pred_values, y=miss_y_pred_counts, x_label="target", y_label="count", title="miss_y_pred", ax=ax[1, 1]
    )

    ax12 = ax[1, 2]
    bars = ax12.bar(miss_y_true_values, miss_y_true_counts, alpha=0.5, label="miss_y_true")
    ax12.bar_label(bars, label_type="center")
    bars = ax12.bar(miss_y_pred_values, miss_y_pred_counts, alpha=0.5, label="miss_y_pred")
    ax12.bar_label(bars)
    ax12.set_title("miss_y_true vs miss_y_pred")
    ax12.set_xticks(miss_y_pred_values)
    ax12.set(xlabel="target", ylabel="count", xticklabels=list(map(str, miss_y_true_values)))
    ax12.legend()

    fig.suptitle("Distribution of y_true and y_pred")
    fig.tight_layout()
    fig.savefig(save_fp)
    plt.close("all")


def heatmap(
    *,
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    x_label: str,
    y_label: str,
    save_fp: pathlib.Path | None = None,
) -> None:
    """plot heatmap

    Args:
        x: array of x-axis
        y: array of y-axis
        x_label: x-axis label
        y_label: y-axis label

    Returns:
        None
    """
    if not issubclass(x.dtype.type, np.integer):
        class_map_x = {class_: i for i, class_ in enumerate(np.unique(x))}
    else:
        class_map_x = None

    if not issubclass(y.dtype.type, np.integer):
        class_map_y = {class_: i for i, class_ in enumerate(np.unique(y))}
    else:
        class_map_y = None

    print(f"class_map_x: {class_map_x}, class_map_y: {class_map_y}")

    heatmap = utils.to_heatmap(x=x, y=y)
    max_size = np.max(heatmap.shape)
    fig, ax = plt.subplots(1, 1, figsize=(2 * max_size, 2 * max_size))
    sns.heatmap(heatmap, annot=True, ax=ax, cmap="Blues", fmt=".0f")
    ax.set(xlabel=x_label, ylabel=y_label, title=f"{x_label} vs {y_label}")
    if class_map_x is not None:
        ax.set_xticklabels(list(map(str, class_map_x.keys())))
    if class_map_y is not None:
        ax.set_yticklabels(list(map(str, class_map_y.keys())))
    fig.tight_layout()
    fig.show()
    if save_fp is not None:
        fig.savefig(save_fp)


def scatter(
    *,
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    x_label: str,
    y_label: str,
    fig_size: tuple[int, int] = (15, 10),
    save_fp: pathlib.Path | None = None,
    ax: axes.Axes | None = None,
) -> None:
    """plot scatter

    Args:
        x: array of x-axis
        y: array of y-axis
        x_label: x-axis label
        y_label: y-axis label
        fig_size: figure size. [Width, Height]
        save_fp: save file path
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = None
    if ax is None:
        raise ValueError("ax is None")
    ax.scatter(x, y, alpha=0.7, marker="x", linewidths=1, label=f"{x_label} vs {y_label}")
    ax.set(xlabel=x_label, ylabel=y_label, title=f"{x_label} vs {y_label}")
    if fig is not None:
        fig.tight_layout()
        fig.show()
    if save_fp is not None and fig is not None:
        fig.savefig(save_fp)


# -------
# 軌跡の可視化
# Ref:
# 1. https://www.guruguru.science/competitions/25/discussions/b75b30bb-abcd-482d-b43a-fa325748e48d/
# -------


def camera_to_image(
    p_camera: npt.NDArray[np.float32], intrinsic_matrix: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Convert camera coordinates to image coordinates

    Args:
        p_camera (np.ndarray): camera coordinates
        intrinsic_matrix (np.ndarray): intrinsic matrix
    """
    p_image_homogenous = np.dot(intrinsic_matrix, p_camera)
    p_image = p_image_homogenous[:2] / p_image_homogenous[2]
    if not isinstance(p_image, np.ndarray):
        raise ValueError(f"p_image is not np.ndarray: {p_image}")
    return p_image


INTRINSIC_MATRIX: Final[npt.NDArray[np.float32]] = np.array(
    [[226.16438356, 0.0, 63.62426614], [0.0, 224.82352941, 11.76], [0.0, 0.0, 1.0]], dtype=np.float32
)


def project_trajectory_to_image_coordinate_system(
    trajectory: npt.NDArray[np.float32], intrinsic_matrix: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Convert camera coordinates to image coordinates

    Args:
        trajectory (np.ndarray): camera coordinates
        intrinsic_matrix (np.ndarray): intrinsic matrix
    """
    tranjectory_with_offset = trajectory.copy()
    # 1.22m offset for the height of the camera
    tranjectory_with_offset[:, 2] = tranjectory_with_offset[:, 2] + 1.22

    # Convert camera coordinates to image coordinates
    # shape: (3, 3)
    road_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32)
    trajectory_camera = tranjectory_with_offset @ road_to_camera
    trajectory_image = np.array(
        [camera_to_image(p_camera, intrinsic_matrix) for p_camera in trajectory_camera if p_camera[2] > 0],
    )
    return trajectory_image


def overlay_trajectory(
    trajectory: npt.NDArray[np.float32] | Iterable[tuple[npt.NDArray[np.float32], str, str]],
    image: npt.NDArray[np.int32],
    title: str,
    figsize: tuple[int, int] = (15, 10),
    fig_lim: tuple[int, int] = (128, 64),
    save_fp: pathlib.Path | None = None,
) -> None:
    """Overlay trajectory on the image

    Args:
        trajectory (np.ndarray): trajectory or list of (trajectory, color, label)
        image (np.ndarray): image
        intrinsic_matrix (np.ndarray): intrinsic matrix
        figsize (tuple, optional): figure size. Defaults to (15, 10).
        fig_lim (tuple, optional): figure limit. Defaults to (xlim, ylim) = (128, 64).
        save_fp (pathlib.Path, optional): save file path. Defaults to None. if None, not save.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.imshow(image)

    if isinstance(trajectory, np.ndarray):
        trajectory_image = project_trajectory_to_image_coordinate_system(trajectory, INTRINSIC_MATRIX)
        ax.plot(
            trajectory_image[:, 0],
            trajectory_image[:, 1],
            marker="o",
            color="forestgreen",
            alpha=1.0,
            markersize=3,
            linestyle="solid",
        )
    else:
        for traj, color, label in trajectory:
            trajectory_image = project_trajectory_to_image_coordinate_system(traj, INTRINSIC_MATRIX)
            print(f"trajectory_image: {trajectory_image.shape}")
            if trajectory_image.shape[0] < 1:
                print(f"Skip {label}, because trajectory_image.shape[0] < 1")
                continue
            ax.plot(
                trajectory_image[:, 0],
                trajectory_image[:, 1],
                marker="o",
                color=color,
                alpha=1.0,
                markersize=3,
                linestyle="solid",
                label=label,
            )
    ax.set_xlim(0, fig_lim[0])
    ax.set_ylim(fig_lim[1], 0)
    ax.set_title(title)
    if not isinstance(trajectory, np.ndarray):
        ax.legend()
    plt.show()
    if save_fp is not None:
        fig.savefig(save_fp)
    plt.close("all")
