import argparse

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb

from src import constants, log, metrics, utils

from . import config, preprocess

plt.style.use("fivethirtyeight")

logger = log.get_root_logger()
CALLED_TIME = log.get_called_time()
COMMIT_HASH = utils.get_commit_hash_head()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-fold", action="store_true")
    return parser.parse_args()


def fill_null(df: pl.DataFrame, fill_num_value: float = -9999, fill_str_value: str = "missing") -> pl.DataFrame:
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            df = df.with_columns(pl.col(col).fill_null(fill_num_value))
        elif dtype in [pl.Utf8, pl.String, pl.Categorical]:
            df = df.with_columns(pl.col(col).fill_null(fill_str_value))
    return df


def main() -> None:
    args = get_args()
    cfg = config.GBDTConfig(seed=args.seed, is_debug=args.debug)
    fname = __file__.split("/")[-1].split(".")[0]
    save_dir = cfg.output_dir / f"{fname}/{cfg.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log.attach_file_handler(logger, str(save_dir / "train_gbdt.log"))

    # --- CV
    if args.make_fold:
        df = pl.read_csv(cfg.train_data_fp)
        # --- Preprocess
        preprocessor = preprocess.Preprocessor()
        df = preprocessor.fit_transform(df)
        # seedで順番変えても良いかもしれん
        df = df.with_columns(fold=pl.lit(-1)).with_row_index()
        for i, scene_id in enumerate(df["scene_id"].unique().sort()):
            df = df.with_columns(
                fold=pl.when(pl.col("scene_id") == scene_id).then(pl.lit(i % cfg.n_folds)).otherwise(pl.col("fold"))
            )
        df.write_parquet(constants.INPUT_DIR / "train_folds.parquet")
    else:
        df = pl.read_parquet(constants.INPUT_DIR / "train_folds.parquet")

    logger.info(f"{cfg.train_data_fp = }")
    utils.pinfo(cfg.model_dump())

    # --- Train
    oof_total = pl.DataFrame()
    for fold in range(cfg.n_folds):
        df_train = df.filter(pl.col("fold") != fold)
        df_valid = df.filter(pl.col("fold") == fold)
        if args.debug:
            df_train = df_train.head(200)
            df_valid = df_valid.head(200)

        target_col = constants.TARGET_COLS
        drop_cols = [*target_col, "ID", "fold", "scene_id", "scene_time", "index"]
        feature_cols = sorted(list(set(df_train.columns) - set(drop_cols)))

        # testの時には列の並びに注意するためにログに出力しておく
        logger.info(f"""
        ####################### Fold: {fold} ############################

        df_train: {df_train}\n  df_valid: {df_valid}

        """)
        utils.pinfo(feature_cols)
        logger.info(f"{len(feature_cols) = }")

        y_train = df_train[target_col].to_numpy()
        ds_train = xgb.DMatrix(
            df_train.select(feature_cols).to_pandas(),
            label=y_train,
            feature_names=feature_cols,
            enable_categorical=True,
            nthread=-1,
            missing=np.inf,
        )
        ds_valid = xgb.DMatrix(
            df_valid.select(feature_cols).to_pandas(),
            label=df_valid[target_col].to_numpy(),
            feature_names=feature_cols,
            enable_categorical=True,
            nthread=-1,
            missing=np.inf,
        )

        model = xgb.train(
            params={
                **cfg.gbdt_model_params,
            },
            dtrain=ds_train,
            num_boost_round=cfg.num_boost_round,
            evals=[(ds_train, "train"), (ds_valid, "valid")],
            verbose_eval=500,
            maximize=cfg.maximize,
        )

        # save feature importances
        importances = model.get_score(importance_type="total_gain")
        df_importances = pl.DataFrame({"feature": importances.keys(), "gain": importances.values()})
        df_importances.write_parquet(save_dir / f"importances_{fold}.parquet")
        utils.save_importances(df_importances, save_dir / f"importances_{fold}.png")
        model.save_model(save_dir / f"xgb_model_{fold}.ubj")

        y_preds = model.predict(ds_valid)
        oof = pl.concat(
            [
                df_valid,
                pl.DataFrame({
                    **{f"pred-{col}": y_preds[:, i] for i, col in enumerate(target_col)},
                }),
            ],
            how="horizontal",
        )
        oof_total = pl.concat([oof_total, oof], how="vertical")

        score_each_dim = {}
        for i, col in enumerate(target_col):
            score_each_dim[col] = metrics.score(y_pred=y_preds[:, i], y_true=df_valid[col].to_numpy())
        score_value = metrics.score(y_pred=y_preds, y_true=df_valid[target_col].to_numpy())

        oof.write_parquet(save_dir / f"oof_{fold}.parquet")
        logger.info(f"{oof_total = }")
        utils.pinfo(score_each_dim)
        logger.info(f"Score: {score_value = }")

    total_score = metrics.score(
        y_pred=oof_total[[f"pred-{c}" for c in target_col]].to_numpy(), y_true=oof_total[target_col].to_numpy()
    )
    logger.info(f"Total Score: {total_score = }")


if __name__ == "__main__":
    main()
