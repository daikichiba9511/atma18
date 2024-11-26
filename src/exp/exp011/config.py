import pathlib
from typing import Any, Literal, TypeAlias

import albumentations as albu
import cv2
import pydantic
import torch
from albumentations.pytorch import ToTensorV2

from src import constants

EXP_NO = __file__.split("/")[-2]
DESCRIPTION = """
simple baseline
"""

AlbuTransforms: TypeAlias = list[albu.BaseCompose | albu.BasicTransform | albu.OneOf | albu.DualTransform]


class Config(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True, protected_namespaces=())
    name: str = EXP_NO
    description: str = DESCRIPTION

    # -- General
    is_debug: bool = False
    root_dir: pathlib.Path = constants.ROOT
    """Root directory. alias to constants.ROOT"""
    input_dir: pathlib.Path = constants.INPUT_DIR
    """input directory. alias to constants.INPUT_DIR"""
    output_dir: pathlib.Path = constants.OUTPUT_DIR / EXP_NO
    """output directory. constants.OUTPUT_DIR/EXP_NO"""
    data_dir: pathlib.Path = constants.DATA_DIR
    """data directory. alias to constants.DATA_DIR"""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0 if is_debug else 16
    seed: int = 42

    # -- Train
    train_log_interval: int = 1
    train_batch_size: int = 32
    train_n_epochs: int = 20

    train_use_amp: bool = True

    train_loss_name: str = "L1Loss"
    train_loss_params: dict[str, str] = {"reduction": "mean"}
    train_optimizer_name: str = "AdamW"
    train_optimizer_params: dict[str, float] = {"lr": 1e-3, "weight_decay": 1e-6, "eps": 1e-8, "fused": False}
    train_scheduler_name: Literal["CosineLRScheduler"] = "CosineLRScheduler"
    train_scheduler_params: dict[str, float] = {
        "num_warmup_steps": 1,
        "num_training_steps": -1,
        "num_cycles": 0.5,
        "last_epoch": -1,
    }

    # -- Valid
    valid_batch_size: int = 32 * 2

    # -- Data
    n_folds: int = 5
    train_data_fp: pathlib.Path = constants.INPUT_DIR / "train_folds.parquet"
    test_data_fp: pathlib.Path = constants.DATA_DIR / "test_features.csv"

    # -- Model
    model_name: str = "Atma18VisionModel"
    model_params: dict[str, str | float | bool | int] = {
        # "model_name": "tf_efficientnet_b3.ns_jft_in1k",
        # "model_name": "resnet34d",
        # "model_name": "convnext_tiny.fb_in22k_ft_in1k",
        "model_name": "convnext_base.fb_in22k_ft_in1k",
        # "model_name": "convnext_large.fb_in22k_ft_in1k",
        # "model_name": "eva02_base_patch14_224.mim_in22k",
        "pretrained": True,
        "in_features": 172,
    }
    cols_aux: tuple[str, ...] = (
        "vEgo",
        "aEgo",
        "steeringAngleDeg",
        # "steeringTorque",
    )
    cols_aux_cls: tuple[str, ...] = (
        "brakePressed",
        "leftBlinker",
        "rightBlinker",
    )
    size: int = 224
    train_tranforms: list = [
        albu.Resize(size, size),
        albu.OneOf([
            albu.GaussNoise(var_limit=(10, 50)),
            albu.GaussianBlur(),
            albu.MotionBlur(),
        ]),
        albu.OneOf([
            albu.RandomGamma(gamma_limit=(30, 150), p=1),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=1),
            albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            albu.CLAHE(clip_limit=5.0, tile_grid_size=(5, 5), p=1),
        ]),
        # albu.HorizontalFlip(p=0.5),
        # albu.ShiftScaleRotate(
        #     shift_limit=0.0,
        #     scale_limit=0.1,
        #     rotate_limit=15,
        #     interpolation=cv2.INTER_LINEAR,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     p=0.8,
        # ),
        albu.CoarseDropout(max_height=50, max_width=50, min_holes=2, p=0.5),
        albu.Normalize(mean=[0] * 9, std=[1] * 9),
        ToTensorV2(),
    ]
    valid_tranforms: list = [
        albu.Resize(size, size),
        albu.Normalize(mean=[0] * 9, std=[1] * 9),
        ToTensorV2(),
    ]


class GBDTConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True, protected_namespaces=())
    name: str = EXP_NO
    description: str = """
    simple baseline
    """
    # -- General
    is_debug: bool = False
    root_dir: pathlib.Path = constants.ROOT
    """Root directory. alias to constants.ROOT"""
    input_dir: pathlib.Path = constants.INPUT_DIR
    """input directory. alias to constants.INPUT_DIR"""
    output_dir: pathlib.Path = constants.OUTPUT_DIR / EXP_NO
    """output directory. constants.OUTPUT_DIR/EXP_NO"""
    data_dir: pathlib.Path = constants.DATA_DIR
    """data directory. alias to constants.DATA_DIR"""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    # -- Train
    train_log_interval: int = 1
    gbdt_model_params: dict[str, Any] = {
        "learning_rate": 0.01,
        "verbosity": -1,
        "seed": seed,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "max_depth": 5,
        "num_leaves": int(0.7 * (2**5)),
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "drop_seed": seed,
        "metric": "mae",
        "objective": "regression_l1",
    }
    num_boost_round: int = 10000
    # num_boost_round: int = 3
    maximize: bool = False
    use_cols: tuple[str, ...] = (
        "vEgo",
        "aEgo",
        "steeringAngleDeg",
        "steeringTorque",
        "brake",
        "brakePressed",
        "gas",
        "gasPressed",
        "leftBlinker",
        "rightBlinker",
        *[f"nn-pred-{c}" for c in constants.TARGET_COLS],  # prediciton by CNN
    )

    # -- Data
    n_folds: int = 5
    train_data_fp: pathlib.Path = constants.DATA_DIR / "train_features.csv"
    test_data_fp: pathlib.Path = constants.DATA_DIR / "test_features.csv"


class PPConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True, protected_namespaces=())
    name: str = EXP_NO
    # -- General
    is_debug: bool = False
    root_dir: pathlib.Path = constants.ROOT
    """Root directory. alias to constants.ROOT"""
    input_dir: pathlib.Path = constants.INPUT_DIR
    """input directory. alias to constants.INPUT_DIR"""
    output_dir: pathlib.Path = constants.OUTPUT_DIR / EXP_NO
    """output directory. constants.OUTPUT_DIR/EXP_NO"""
    data_dir: pathlib.Path = constants.DATA_DIR
    """data directory. alias to constants.DATA_DIR"""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    # -- Train
    train_log_interval: int = 1
    gbdt_model_params: dict[str, Any] = {
        "learning_rate": 0.01,
        "verbosity": -1,
        "seed": seed,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "max_depth": 5,
        "num_leaves": int(0.7 * (2**5)),
        "bagging_seed": seed,
        "bagging_fraction": 0.3,
        "feature_fraction_seed": seed,
        "feature_fraction": 0.3,
        "drop_seed": seed,
        "metric": "mae",
        "objective": "regression_l1",
    }
    num_boost_round: int = 2000
    # num_boost_round: int = 3
    maximize: bool = False
    use_cols: tuple[str, ...] = (
        "vEgo",
        "aEgo",
        "steeringAngleDeg",
        "steeringTorque",
        "brake",
        "brakePressed",
        "gas",
        "gasPressed",
        "leftBlinker",
        "rightBlinker",
        *[f"nn-pred-{c}" for c in constants.TARGET_COLS],  # prediciton by CNN
    )

    # -- Data
    n_folds: int = 5
    train_data_fp: pathlib.Path = constants.DATA_DIR / "train_features.csv"
    test_data_fp: pathlib.Path = constants.DATA_DIR / "test_features.csv"

    oof_nn_paths: list[pathlib.Path] = [
        constants.OUTPUT_DIR / name / "oof_0.parquet",
        constants.OUTPUT_DIR / name / "oof_1.parquet",
        constants.OUTPUT_DIR / name / "oof_2.parquet",
        constants.OUTPUT_DIR / name / "oof_3.parquet",
        constants.OUTPUT_DIR / name / "oof_4.parquet",
    ]

    oof_gbdt_paths: list[pathlib.Path] = [
        constants.OUTPUT_DIR / name / "train_gbdt" / f"{seed}" / "oof_0.parquet",
        constants.OUTPUT_DIR / name / "train_gbdt" / f"{seed}" / "oof_1.parquet",
        constants.OUTPUT_DIR / name / "train_gbdt" / f"{seed}" / "oof_2.parquet",
        constants.OUTPUT_DIR / name / "train_gbdt" / f"{seed}" / "oof_3.parquet",
        constants.OUTPUT_DIR / name / "train_gbdt" / f"{seed}" / "oof_4.parquet",
    ]
