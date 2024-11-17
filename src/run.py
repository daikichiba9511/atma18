import pathlib
from typing import Iterable

import numpy as np
import polars as pl
import xgboost as xgb

from src import constants, log, metrics, utils


def get_models(model_fps: Iterable[pathlib.Path] | None = None) -> list[tuple[float, xgb.Booster]]:
    if model_fps is None:
        model_fps = [
            constants.OUTPUT_DIR / "exp000" / "train_gbdt" / "42" / "xgb_model_0.xgb",
            constants.OUTPUT_DIR / "exp000" / "train_gbdt" / "42" / "xgb_model_1.xgb",
            constants.OUTPUT_DIR / "exp000" / "train_gbdt" / "42" / "xgb_model_2.xgb",
            constants.OUTPUT_DIR / "exp000" / "train_gbdt" / "42" / "xgb_model_3.xgb",
            constants.OUTPUT_DIR / "exp000" / "train_gbdt" / "42" / "xgb_model_4.xgb",
        ]

    models = []
    for model_fp in model_fps:
        model = xgb.Booster()
        model.load_model(str(model_fp))
        models.append((1 / 5, model))
    return models


def infer(df: pl.DataFrame) -> dict[str, np.ndarray]:
    # --- Preprocess
    from src.exp.exp000 import preprocess

    preprocessor = preprocess.Preprocessor()
    df = preprocessor.transform(df)
    target_col = constants.TARGET_COLS
    drop_cols = [*target_col, "ID", "fold", "scene_id", "scene_time", "index"]
    feature_cols = sorted(list(set(df.columns) - set(drop_cols)))
    utils.pinfo(feature_cols)

    # --- Infer
    df_infer = df.select(feature_cols)
    ds_infer = xgb.DMatrix(
        df_infer.to_pandas(),
        feature_names=feature_cols,
        enable_categorical=True,
        nthread=-1,
        missing=np.inf,
    )
    models = get_models()
    y_preds = np.zeros((len(models), df.shape[0], df.shape[1]))
    for i_model, (_w, model) in enumerate(models):
        y_pred = model.predict(ds_infer)
        y_preds[i_model, :, :] = y_pred

    # --- Ensemble
    y_pred = y_preds.mean(axis=0)
    return {"y_pred": y_pred}


def run_valid() -> None:
    logger = log.get_root_logger()
    df = pl.read_parquet(constants.INPUT_DIR / "train_folds.parquet")
    oof = pl.DataFrame()
    for fold in range(5):
        df_valid = df.filter(pl.col("fold") == fold)
        out = infer(df_valid)
        y_pred = out["y_pred"]
        oof = pl.concat([
            oof,
            pl.concat([
                df_valid,
                pl.DataFrame({f"pred-{c}": y_pred[:, i_c] for i_c, c in enumerate(constants.TARGET_COLS)}),
            ]),
        ])
    print(oof)
    metric_score = metrics.score(
        oof[constants.TARGET_COLS].to_numpy(), oof[[f"pred-{c}" for c in constants.TARGET_COLS]].to_numpy()
    )
    logger.info(f"metric_score: {metric_score}")


if __name__ == "__main__":
    run_valid()
