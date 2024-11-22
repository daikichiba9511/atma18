import argparse
from datetime import datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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


def main() -> None:
    args = get_args()
    cfg = config.GBDTConfig(seed=args.seed, is_debug=args.debug)
    fname = __file__.split("/")[-1].split(".")[0]
    save_dir = cfg.output_dir / f"{fname}/{cfg.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log.attach_file_handler(logger, str(save_dir / "train_gbdt.log"))

    # --- CV
    if args.make_fold:
        df_train = pl.read_csv(cfg.train_data_fp)
        df_test = pl.read_csv(cfg.test_data_fp)
        # --- Traffic Light
        df_traffic_light = preprocess.load_df_traffic_light(constants.DATA_DIR / "traffic_lights")
        df_traffic_light_count = preprocess.make_df_traffic_light_count(df_traffic_light)
        df_traffic_col = [c for c in df_traffic_light_count.columns if c not in ["ID"]]
        # --- Read OOF prediction by CNN, and then join it with df
        oof_pred_cols = [f"pred-{c}" for c in constants.TARGET_COLS]
        oof_nn = (
            pl.read_parquet(constants.OUTPUT_DIR / cfg.name / "oof.parquet", columns=["sample_id", *oof_pred_cols])
            .with_columns(ID=pl.col("sample_id").str.split("/").list[-1])
            .rename({c: f"nn-{c}" for c in oof_pred_cols})
        )
        # --- Preprocess
        common_col = sorted(list(set(df_train.columns) & set(df_test.columns)))
        df_all = pl.concat([df_train.select(common_col), df_test.select(common_col)])
        df_all = df_all.join(df_traffic_light_count, on="ID", how="left").with_columns(*[
            pl.col(c).fill_null(0).alias(c) for c in df_traffic_col
        ])
        df_all = df_all.join(oof_nn, on="ID", how="left").with_columns(*[
            pl.col(c).fill_null(0).alias(c) for c in oof_nn.columns if c not in ["sample_id", "ID"]
        ])
        df_all = preprocess.cast_dtype(df_all)
        print(f"{df_all = }")
        df_all = preprocess.add_feature(df_all, cfg.use_cols)

        df = df_train.select(["ID", *constants.TARGET_COLS]).join(df_all, on="ID", how="left")
        # df_test = df_test.join(df_all, on="ID", how="left")

        # seedで順番変えても良いかもしれん
        df = df.with_columns(fold=pl.lit(-1)).with_row_index()
        for i, scene_id in enumerate(df["scene_id"].unique().sort()):
            df = df.with_columns(
                fold=pl.when(pl.col("scene_id") == scene_id).then(pl.lit(i % cfg.n_folds)).otherwise(pl.col("fold"))
            )
        df.write_parquet(constants.INPUT_DIR / "train_folds.parquet")
    else:
        df = pl.read_parquet(constants.INPUT_DIR / "train_folds.parquet")
    df = utils.reduce_memory_usage_pl(df, name="df")
    logger.info(f"{df = }")
    logger.info(f"{cfg.train_data_fp = }")
    utils.pinfo(cfg.model_dump())

    # --- Train
    oof_total = pl.DataFrame()
    scores_total = []
    for fold in range(cfg.n_folds):
        df_train = df.filter(pl.col("fold") != fold)
        df_valid = df.filter(pl.col("fold") == fold)
        if args.debug:
            df_train = df_train.head(200)
            df_valid = df_valid.head(200)

        target_cols = constants.TARGET_COLS
        drop_cols = [*target_cols, "ID", "fold", "scene_id", "scene_time", "index"]
        feature_cols = sorted(list(set(df_train.columns) - set(drop_cols)))
        category_cols = [c for c in feature_cols if df_train[c].dtype == pl.Categorical]

        # testの時には列の並びに注意するためにログに出力しておく
        logger.info(f"""
        ####################### Fold: {fold} ############################

        df_train: {df_train}\n  df_valid: {df_valid}

        """)
        utils.pinfo(feature_cols)
        logger.info(f"{len(feature_cols) = }")
        y_train = df_train[target_cols].to_numpy()
        y_valid = df_valid[target_cols].to_numpy()

        y_pred_total = np.zeros((len(df_valid), len(target_cols)))
        for i_col, col in enumerate(target_cols):
            logger.info(f"Training {col} / Fold: {fold}")
            ds_train = lgb.Dataset(
                df_train.select(feature_cols).to_pandas(),
                label=y_train[:, i_col],
                feature_name=feature_cols,
                categorical_feature=category_cols,
                free_raw_data=False,
            )
            ds_valid = lgb.Dataset(
                df_valid.select(feature_cols).to_pandas(),
                label=y_valid[:, i_col],
                reference=ds_train,
                feature_name=feature_cols,
                categorical_feature=category_cols,
                free_raw_data=False,
            )

            with utils.trace(f"Train Fold: {fold}"):
                logger.info(f"params: {cfg.gbdt_model_params}")
                model = lgb.train(
                    params={
                        **cfg.gbdt_model_params,
                    },
                    train_set=ds_train,
                    num_boost_round=cfg.num_boost_round,
                    valid_sets=[ds_train, ds_valid],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.callback.log_evaluation(100)],
                )

            # save feature importances
            importances = model.feature_importance(importance_type="gain")
            feature_cols_model = model.feature_name()
            df_importances = pl.DataFrame({"feature": feature_cols_model, "gain": importances})
            df_importances.write_parquet(save_dir / f"importances_{fold}_{col}.parquet")
            utils.save_importances(df_importances, save_dir / f"importances_{fold}_{col}.png")
            model.save_model(save_dir / f"xgb_model_{fold}_{col}.ubj")
            y_preds = model.predict(ds_valid.data)
            y_pred_total[:, i_col] = y_preds

        oof = pl.concat(
            [
                df_valid,
                pl.DataFrame({
                    **{f"pred-{col}": y_pred_total[:, i] for i, col in enumerate(target_cols)},
                }),
            ],
            how="horizontal",
        )
        oof_total = pl.concat([oof_total, oof], how="vertical")

        score_each_dim = {}
        for i, col in enumerate(target_cols):
            score_each_dim[col] = metrics.score(y_pred=oof[f"pred-{col}"].to_numpy(), y_true=oof[col].to_numpy())
        score_value = metrics.score(
            y_pred=oof[[f"pred-{col}" for col in target_cols]].to_numpy(), y_true=df_valid[target_cols].to_numpy()
        )
        scores_total.append(score_value)

        oof.write_parquet(save_dir / f"oof_{fold}.parquet")
        logger.info(f"{oof_total = }")
        utils.pinfo(score_each_dim)
        logger.info(f"Score: {score_value = }")

    total_score = metrics.score(
        y_pred=oof_total[[f"pred-{c}" for c in target_cols]].to_numpy(), y_true=oof_total[target_cols].to_numpy()
    )
    logger.info(f"""
==========================
Exp: {cfg.name}, DESC: {cfg.description}

Total Score: {total_score = }

Scores: {scores_total}
Mean: {np.mean(scores_total)} +/- {np.std(scores_total)}

Training finished. {CALLED_TIME = }, {COMMIT_HASH = }, Duration: {log.calc_duration_from(CALLED_TIME)}
==========================
    """)


if __name__ == "__main__":
    main()
