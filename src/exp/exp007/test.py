import multiprocessing as mp
import pathlib
from typing import Mapping, TypeAlias

import albumentations as albu
import cv2
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from tqdm.auto import tqdm

from src import constants, log, utils

from . import config, models, preprocess

DO_PP = False
logger = log.get_root_logger()
EXP_NO = __file__.split("/")[-2]
CALLED_TIME = log.get_called_time()
COMMIT_HASH = utils.get_commit_hash_head()
FEATURE_COLS = [
    "aEgo",
    "aEgo_diff+1",
    "aEgo_diff-1",
    "aEgo_max",
    "aEgo_mean",
    "aEgo_median",
    "aEgo_min",
    "aEgo_shift+1",
    "aEgo_shift-1",
    "aEgo_std",
    "brake",
    "brakePressed",
    "brakePressed_diff+1",
    "brakePressed_diff-1",
    "brakePressed_max",
    "brakePressed_mean",
    "brakePressed_median",
    "brakePressed_min",
    "brakePressed_shift+1",
    "brakePressed_shift-1",
    "brakePressed_std",
    "brakePressed_sum",
    "brake_diff+1",
    "brake_diff-1",
    "brake_max",
    "brake_mean",
    "brake_mean_right",
    "brake_median",
    "brake_min",
    "brake_shift+1",
    "brake_shift-1",
    "brake_std",
    "empty",
    "fast",
    "fast_acc",
    "gas",
    "gasPressed",
    "gasPressed_diff+1",
    "gasPressed_diff-1",
    "gasPressed_max",
    "gasPressed_mean",
    "gasPressed_median",
    "gasPressed_min",
    "gasPressed_shift+1",
    "gasPressed_shift-1",
    "gasPressed_std",
    "gasPressed_sum",
    "gas_diff+1",
    "gas_diff-1",
    "gas_max",
    "gas_mean",
    "gas_median",
    "gas_min",
    "gas_shift+1",
    "gas_shift-1",
    "gas_std",
    "gearShifter",
    "green",
    "harsh_acceleration_sum",
    "harsh_braking_sum",
    "left",
    "leftBlinker",
    "leftBlinker_diff+1",
    "leftBlinker_diff-1",
    "leftBlinker_max",
    "leftBlinker_mean",
    "leftBlinker_median",
    "leftBlinker_min",
    "leftBlinker_shift+1",
    "leftBlinker_shift-1",
    "leftBlinker_std",
    "leftBlinker_sum",
    "nn-pred-x_0",
    "nn-pred-x_0_diff+1",
    "nn-pred-x_0_diff-1",
    "nn-pred-x_0_max",
    "nn-pred-x_0_mean",
    "nn-pred-x_0_median",
    "nn-pred-x_0_min",
    "nn-pred-x_0_shift+1",
    "nn-pred-x_0_shift-1",
    "nn-pred-x_0_std",
    "nn-pred-x_1",
    "nn-pred-x_1_diff+1",
    "nn-pred-x_1_diff-1",
    "nn-pred-x_1_max",
    "nn-pred-x_1_mean",
    "nn-pred-x_1_median",
    "nn-pred-x_1_min",
    "nn-pred-x_1_shift+1",
    "nn-pred-x_1_shift-1",
    "nn-pred-x_1_std",
    "nn-pred-x_2",
    "nn-pred-x_2_diff+1",
    "nn-pred-x_2_diff-1",
    "nn-pred-x_2_max",
    "nn-pred-x_2_mean",
    "nn-pred-x_2_median",
    "nn-pred-x_2_min",
    "nn-pred-x_2_shift+1",
    "nn-pred-x_2_shift-1",
    "nn-pred-x_2_std",
    "nn-pred-x_3",
    "nn-pred-x_3_diff+1",
    "nn-pred-x_3_diff-1",
    "nn-pred-x_3_max",
    "nn-pred-x_3_mean",
    "nn-pred-x_3_median",
    "nn-pred-x_3_min",
    "nn-pred-x_3_shift+1",
    "nn-pred-x_3_shift-1",
    "nn-pred-x_3_std",
    "nn-pred-x_4",
    "nn-pred-x_4_diff+1",
    "nn-pred-x_4_diff-1",
    "nn-pred-x_4_max",
    "nn-pred-x_4_mean",
    "nn-pred-x_4_median",
    "nn-pred-x_4_min",
    "nn-pred-x_4_shift+1",
    "nn-pred-x_4_shift-1",
    "nn-pred-x_4_std",
    "nn-pred-x_5",
    "nn-pred-x_5_diff+1",
    "nn-pred-x_5_diff-1",
    "nn-pred-x_5_max",
    "nn-pred-x_5_mean",
    "nn-pred-x_5_median",
    "nn-pred-x_5_min",
    "nn-pred-x_5_shift+1",
    "nn-pred-x_5_shift-1",
    "nn-pred-x_5_std",
    "nn-pred-y_0",
    "nn-pred-y_0_diff+1",
    "nn-pred-y_0_diff-1",
    "nn-pred-y_0_max",
    "nn-pred-y_0_mean",
    "nn-pred-y_0_median",
    "nn-pred-y_0_min",
    "nn-pred-y_0_shift+1",
    "nn-pred-y_0_shift-1",
    "nn-pred-y_0_std",
    "nn-pred-y_1",
    "nn-pred-y_1_diff+1",
    "nn-pred-y_1_diff-1",
    "nn-pred-y_1_max",
    "nn-pred-y_1_mean",
    "nn-pred-y_1_median",
    "nn-pred-y_1_min",
    "nn-pred-y_1_shift+1",
    "nn-pred-y_1_shift-1",
    "nn-pred-y_1_std",
    "nn-pred-y_2",
    "nn-pred-y_2_diff+1",
    "nn-pred-y_2_diff-1",
    "nn-pred-y_2_max",
    "nn-pred-y_2_mean",
    "nn-pred-y_2_median",
    "nn-pred-y_2_min",
    "nn-pred-y_2_shift+1",
    "nn-pred-y_2_shift-1",
    "nn-pred-y_2_std",
    "nn-pred-y_3",
    "nn-pred-y_3_diff+1",
    "nn-pred-y_3_diff-1",
    "nn-pred-y_3_max",
    "nn-pred-y_3_mean",
    "nn-pred-y_3_median",
    "nn-pred-y_3_min",
    "nn-pred-y_3_shift+1",
    "nn-pred-y_3_shift-1",
    "nn-pred-y_3_std",
    "nn-pred-y_4",
    "nn-pred-y_4_diff+1",
    "nn-pred-y_4_diff-1",
    "nn-pred-y_4_max",
    "nn-pred-y_4_mean",
    "nn-pred-y_4_median",
    "nn-pred-y_4_min",
    "nn-pred-y_4_shift+1",
    "nn-pred-y_4_shift-1",
    "nn-pred-y_4_std",
    "nn-pred-y_5",
    "nn-pred-y_5_diff+1",
    "nn-pred-y_5_diff-1",
    "nn-pred-y_5_max",
    "nn-pred-y_5_mean",
    "nn-pred-y_5_median",
    "nn-pred-y_5_min",
    "nn-pred-y_5_shift+1",
    "nn-pred-y_5_shift-1",
    "nn-pred-y_5_std",
    "nn-pred-z_0",
    "nn-pred-z_0_diff+1",
    "nn-pred-z_0_diff-1",
    "nn-pred-z_0_max",
    "nn-pred-z_0_mean",
    "nn-pred-z_0_median",
    "nn-pred-z_0_min",
    "nn-pred-z_0_shift+1",
    "nn-pred-z_0_shift-1",
    "nn-pred-z_0_std",
    "nn-pred-z_1",
    "nn-pred-z_1_diff+1",
    "nn-pred-z_1_diff-1",
    "nn-pred-z_1_max",
    "nn-pred-z_1_mean",
    "nn-pred-z_1_median",
    "nn-pred-z_1_min",
    "nn-pred-z_1_shift+1",
    "nn-pred-z_1_shift-1",
    "nn-pred-z_1_std",
    "nn-pred-z_2",
    "nn-pred-z_2_diff+1",
    "nn-pred-z_2_diff-1",
    "nn-pred-z_2_max",
    "nn-pred-z_2_mean",
    "nn-pred-z_2_median",
    "nn-pred-z_2_min",
    "nn-pred-z_2_shift+1",
    "nn-pred-z_2_shift-1",
    "nn-pred-z_2_std",
    "nn-pred-z_3",
    "nn-pred-z_3_diff+1",
    "nn-pred-z_3_diff-1",
    "nn-pred-z_3_max",
    "nn-pred-z_3_mean",
    "nn-pred-z_3_median",
    "nn-pred-z_3_min",
    "nn-pred-z_3_shift+1",
    "nn-pred-z_3_shift-1",
    "nn-pred-z_3_std",
    "nn-pred-z_4",
    "nn-pred-z_4_diff+1",
    "nn-pred-z_4_diff-1",
    "nn-pred-z_4_max",
    "nn-pred-z_4_mean",
    "nn-pred-z_4_median",
    "nn-pred-z_4_min",
    "nn-pred-z_4_shift+1",
    "nn-pred-z_4_shift-1",
    "nn-pred-z_4_std",
    "nn-pred-z_5",
    "nn-pred-z_5_diff+1",
    "nn-pred-z_5_diff-1",
    "nn-pred-z_5_max",
    "nn-pred-z_5_mean",
    "nn-pred-z_5_median",
    "nn-pred-z_5_min",
    "nn-pred-z_5_shift+1",
    "nn-pred-z_5_shift-1",
    "nn-pred-z_5_std",
    "normal",
    "normal_acc",
    "other",
    "red",
    "right",
    "rightBlinker",
    "rightBlinker_diff+1",
    "rightBlinker_diff-1",
    "rightBlinker_max",
    "rightBlinker_mean",
    "rightBlinker_median",
    "rightBlinker_min",
    "rightBlinker_shift+1",
    "rightBlinker_shift-1",
    "rightBlinker_std",
    "rightBlinker_sum",
    "sharp_steering_sum",
    "slow",
    "slow_acc",
    "speed_change_rate_mean",
    "speed_change_rate_min",
    "speed_change_rate_std",
    "steeringAngleDeg",
    "steeringAngleDeg_diff+1",
    "steeringAngleDeg_diff-1",
    "steeringAngleDeg_max",
    "steeringAngleDeg_mean",
    "steeringAngleDeg_median",
    "steeringAngleDeg_min",
    "steeringAngleDeg_shift+1",
    "steeringAngleDeg_shift-1",
    "steeringAngleDeg_std",
    "steeringTorque",
    "steeringTorque_diff+1",
    "steeringTorque_diff-1",
    "steeringTorque_max",
    "steeringTorque_mean",
    "steeringTorque_median",
    "steeringTorque_min",
    "steeringTorque_shift+1",
    "steeringTorque_shift-1",
    "steeringTorque_std",
    "steering_change_rate_mean",
    "steering_change_rate_min",
    "steering_change_rate_std",
    "straight",
    "vEgo",
    "vEgo_diff+1",
    "vEgo_diff-1",
    "vEgo_max",
    "vEgo_mean",
    "vEgo_median",
    "vEgo_min",
    "vEgo_shift+1",
    "vEgo_shift-1",
    "vEgo_std",
    "very_fast",
    "very_fast_acc",
    "very_slow",
    "very_slow_acc",
    "yellow",
]


def read_multiframe_image(
    base_path: pathlib.Path, image_caches: Mapping[str, npt.NDArray] | None
) -> list[npt.NDArray]:
    fps = [str(base_path / "image_t-1.0.png"), str(base_path / "image_t-0.5.png"), str(base_path / "image_t.png")]
    images = []
    for fp in fps:
        if image_caches is not None and fp in image_caches:
            image = image_caches[fp]
        else:
            image = cv2.imread(fp)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


TestBatch: TypeAlias = tuple[str, torch.Tensor, int]
AlbuTransforms: TypeAlias = list[albu.BasicTransform | albu.OneOf | albu.BaseCompose]


def as_compose(composed_fns: AlbuTransforms) -> albu.ReplayCompose:
    return albu.ReplayCompose(composed_fns)


def read_image(fp: str) -> tuple[str, npt.NDArray]:
    if not pathlib.Path(fp).exists():
        raise FileNotFoundError(f"{fp} is not found.")
    img = cv2.imread(fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return fp, img


class MyTestDataset(torch_data.Dataset[TestBatch]):
    def __init__(
        self,
        df: pl.DataFrame,
        transforms: AlbuTransforms | None = None,
        video_caches: Mapping[str, npt.NDArray] | None = None,
    ) -> None:
        super().__init__()
        self.df = df
        self.transform_fn = as_compose(transforms) if transforms is not None else None
        self.video_caches = video_caches

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TestBatch:
        base_path = pathlib.Path(self.df["base_path"][idx])
        image = read_multiframe_image(base_path, self.video_caches)
        if self.transform_fn is not None:
            replay: dict | None = None
            images = []
            for img in image:
                if replay is None:
                    sample = self.transform_fn(image=img)
                    replay = sample["replay"]
                    images.append(sample["image"])
                else:
                    sample = albu.ReplayCompose.replay(replay, image=img)
                    images.append(sample["image"])
            imgs = torch.concat(images, dim=0)
        else:
            imgs = torch.concat([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in image], dim=0)

        return str(base_path), imgs, -999


def init_dataloader(
    df_fp: pathlib.Path,
    batch_size: int,
    num_workers: int = 16,
    transforms: AlbuTransforms | None = None,
    debug: bool = False,
) -> torch_data.DataLoader:
    if mp.cpu_count() < num_workers:
        num_workers = mp.cpu_count()

    if df_fp.suffix == ".csv":
        df = pl.read_csv(df_fp)
    else:
        df = pl.read_parquet(df_fp)

    df = df.with_columns(base_path=str(constants.DATA_DIR) + "/images/" + df["ID"].cast(pl.Utf8) + "/")
    if debug:
        df = df.head(200)

    df_traffic_light = preprocess.load_df_traffic_light(constants.DATA_DIR / "traffic_lights")
    df_traffic_light_count = preprocess.make_df_traffic_light_count(df_traffic_light)
    df_traffic_col = [c for c in df_traffic_light_count.columns if c not in ["ID"]]
    df = df.join(df_traffic_light_count, on="ID", how="left").with_columns(*[
        pl.col(c).fill_null(0).alias(c) for c in df_traffic_col
    ])

    cache_paths = []
    for base_path in df["base_path"].to_list():
        cache_paths.extend([
            f"{base_path}/image_t-1.0.png",
            f"{base_path}/image_t-0.5.png",
            f"{base_path}/image_t.png",
        ])
    video_cached = dict(utils.call_mp_unordered(read_image, cache_paths, with_progress=True))
    ds: torch_data.Dataset[TestBatch] = MyTestDataset(df, video_caches=video_cached, transforms=transforms)

    testloader = torch_data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )
    return testloader


def test_nn(debug: bool = False) -> pl.DataFrame:
    cfg_nn = config.Config()

    # --- Load NN models
    models_ = []
    for fold in range(cfg_nn.n_folds):
        model, _ = models.get_model(cfg_nn.model_name, cfg_nn.model_params)
        model.load_state_dict(torch.load(constants.OUTPUT_DIR / cfg_nn.name / f"last_model_{fold}.pth"))
        models_.append(model.to(cfg_nn.device).eval())

    # --- Inference
    testloader = init_dataloader(
        cfg_nn.test_data_fp,
        cfg_nn.valid_batch_size,
        num_workers=cfg_nn.num_workers,
        transforms=cfg_nn.valid_tranforms,
        debug=debug,
    )
    y_pred = []
    pbar = tqdm(testloader, total=len(testloader), desc="Inference of NN")
    for batch in pbar:
        key, imgs, _ = batch
        imgs = imgs.to(cfg_nn.device)
        logits_total = np.zeros((imgs.size(0), 18, len(models_)))
        with torch.no_grad():
            for i_model, model in enumerate(models_):
                out = model(imgs)
                # shape: (bs, 18) 18 are the num of (x0, y0, z0, ..., x5, y5, z5)
                logits = out["logits"].cpu().numpy().reshape(-1, 18)
                logits_total[:, :, i_model] = logits
        # shape: (bs, 18)
        logits_total = np.mean(logits_total, axis=-1)
        y_pred.append({
            "ID": [k.split("/")[-1] for k in key],
            **{f"nn-pred-{c}": logits_total[:, i] for i, c in enumerate(constants.TARGET_COLS)},
        })
    pred_nn = pl.DataFrame()
    for d in y_pred:
        pred_nn = pl.concat([pred_nn, pl.DataFrame(d)])
    return pred_nn


def main() -> None:
    cfg_gbdt = config.GBDTConfig()

    # --- Load Traffic Light Data
    df_traffic_light = preprocess.load_df_traffic_light(constants.DATA_DIR / "traffic_lights")
    df_traffic_light_count = preprocess.make_df_traffic_light_count(df_traffic_light)
    df_traffic_col = [c for c in df_traffic_light_count.columns if c not in ["ID"]]

    # --- Make Prediction by CNN
    pred_nn = test_nn()

    df_train = pl.read_csv(cfg_gbdt.train_data_fp)
    df_test = pl.read_csv(cfg_gbdt.test_data_fp)
    common_col = sorted(list(set(df_train.columns) & set(df_test.columns)))

    # --- Preprocess
    df_all = (
        pl.concat([df_train.select(common_col), df_test.select(common_col)])
        .join(df_traffic_light_count, on="ID", how="left")
        .with_columns(*[pl.col(c).fill_null(0).alias(c) for c in df_traffic_col])
        .join(pred_nn, on="ID", how="left")
        .with_columns(*[pl.col(c).fill_null(0).alias(c) for c in pred_nn.columns if c not in ["sample_id", "ID"]])
    )
    df_all = preprocess.cast_dtype(df_all)
    df_feats = preprocess.add_feature(df_all, cfg_gbdt.use_cols)

    # --- Make Test Dataset
    df_test = df_test.select(["ID"]).join(df_feats, on="ID", how="left")
    category_cols = [c for c in FEATURE_COLS if df_test[c].dtype == pl.Categorical]
    ds_test = lgb.Dataset(
        df_test.select(FEATURE_COLS).to_pandas(),
        feature_name=FEATURE_COLS,
        categorical_feature=category_cols,
        free_raw_data=False,
    )

    # --- Load GBDT models
    # full trainの時は0だけ読む
    gbdt_models = []
    for fold in [0, 1, 2, 3, 4]:
        gbdt_models.append({
            target_col: lgb.Booster(
                model_file=cfg_gbdt.output_dir / f"train_gbdt/{cfg_gbdt.seed}" / f"xgb_model_{fold}_{target_col}.ubj"
            )
            for target_col in constants.TARGET_COLS
        })

    # --- Make Prediction
    pred_gbdt = np.zeros((len(df_test), len(constants.TARGET_COLS), len(gbdt_models)))
    for i_model, model_dict in enumerate(gbdt_models):
        for i_col, col in enumerate(constants.TARGET_COLS):
            model_: lgb.Booster = model_dict[col]
            pred = model_.predict(ds_test.data)
            assert isinstance(pred, np.ndarray)
            pred_gbdt[:, i_col, i_model] = pred
    pred_gbdt = np.mean(pred_gbdt, axis=-1)
    assert pred_gbdt.shape == (len(df_test), len(constants.TARGET_COLS)), f"{pred_gbdt.shape = }"

    if DO_PP:
        df_pred_gbdt = pl.DataFrame({
            "ID": df_test["ID"],
            **{f"gbdt-{c}": pred_gbdt[:, i] for i, c in enumerate(constants.TARGET_COLS)},
        })
        gbdt_pred_cols = [f"gbdt-{c}" for c in constants.TARGET_COLS]
        df_test = df_test.join(df_pred_gbdt, on="ID", how="left")
        feature_cols = FEATURE_COLS + gbdt_pred_cols
        ds_test = lgb.Dataset(
            df_test.select(feature_cols).to_pandas(),
            feature_name=feature_cols,
            categorical_feature=category_cols,
            free_raw_data=False,
        )
        # --- Load GBDT models
        # full trainの時は0だけ読む
        gbdt_models = []
        for fold in [0, 1, 2, 3, 4]:
            gbdt_models.append({
                target_col: lgb.Booster(
                    model_file=cfg_gbdt.output_dir
                    / f"pp_gbdt/{cfg_gbdt.seed}"
                    / f"pp_gbdt_model_{fold}_{target_col}.ubj"
                )
                for target_col in constants.TARGET_COLS
            })
        pred_gbdt = np.zeros((len(df_test), len(constants.TARGET_COLS), len(gbdt_models)))
        for i_model, model_dict in enumerate(gbdt_models):
            for i_col, col in enumerate(constants.TARGET_COLS):
                model_: lgb.Booster = model_dict[col]
                pred = model_.predict(ds_test.data)
                assert isinstance(pred, np.ndarray)
                pred_gbdt[:, i_col, i_model] = pred
        pred_gbdt = np.mean(pred_gbdt, axis=-1)
        assert pred_gbdt.shape == (len(df_test), len(constants.TARGET_COLS)), f"{pred_gbdt.shape = }"
        pred_gbdt = pl.DataFrame({
            "ID": df_test["ID"],
            **{f"gbdt-pred-{c}": pred_gbdt[:, i] for i, c in enumerate(constants.TARGET_COLS)},
        })
    else:
        pred_gbdt = pl.DataFrame({
            "ID": df_test["ID"],
            **{f"gbdt-pred-{c}": pred_gbdt[:, i] for i, c in enumerate(constants.TARGET_COLS)},
        })
    print(pred_gbdt)
    print(pred_gbdt.columns)

    # --- Save Submission
    sub = (
        pl.read_csv(constants.DATA_DIR / "test_features.csv", columns=["ID"])
        .join(pred_gbdt, on="ID", how="left")
        .rename({c: c.replace("gbdt-pred-", "") for c in pred_gbdt.columns if c != "ID"})
    )
    print(sub)
    save_fp = constants.OUTPUT_DIR / f"{EXP_NO}_{CALLED_TIME}_submission.csv"
    sub.select(constants.TARGET_COLS).write_csv(save_fp)
    print(f"Submission file is saved to {save_fp}")


if __name__ == "__main__":
    main()
