import pathlib

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from src import constants, log, metrics, utils, vis

from . import config, models, preprocess

logger = log.get_root_logger()
CALLED_TIME = log.get_called_time()

cfg = config.PPConfig()
# N_TRIALS = 1
# NUM_BOOST_ROUND = 1

N_TRIALS = 20
NUM_BOOST_ROUND = cfg.num_boost_round


def main() -> None:
    log.attach_file_handler(logger, str(cfg.output_dir / "pp_gbdt.log"))
    utils.pinfo(cfg.model_dump())
    save_dir = cfg.output_dir / "pp_gbdt"
    save_dir.mkdir(parents=True, exist_ok=True)

    df_train = pl.read_csv(cfg.train_data_fp)
    target_cols = constants.TARGET_COLS

    pred_cols = [f"pred-{c}" for c in target_cols]
    nn_pred_cols = [f"nn-{c}" for c in pred_cols]
    gbdt_pred_cols = [f"gbdt-{c}" for c in pred_cols]

    oof_nn = (
        pl.concat([
            pl.read_parquet(fp).with_columns(fold=pl.lit(int(fp.stem.split(".")[-1].split("_")[-1])))
            for fp in cfg.oof_nn_paths
        ])
        .with_columns(
            *[pl.col(c).alias(f"nn-{c}") for c in pred_cols],
            ID=pl.col("sample_id").str.split("/").list[-1],
        )
        .select(["ID", *[f"nn-{c}" for c in pred_cols]])
    )

    oof_gbdt = pl.concat([pl.read_parquet(fp) for fp in cfg.oof_gbdt_paths]).with_columns(
        *[pl.col(c).alias(f"gbdt-{c}") for c in pred_cols],
        ID=pl.col("ID").cast(pl.String),
    )
    print(f"{oof_nn = }")
    print(f"{oof_gbdt = }")

    df_train = (
        df_train.join(oof_nn, on="ID", how="left").join(oof_gbdt, on="ID", how="left")
        # .select(["ID", "fold", *target_cols, *nn_pred_cols, *gbdt_pred_cols])
    )
    df_train = preprocess.cast_dtype(df_train)
    df_feats = preprocess.add_feature(df_train)
    df_train = df_train.select(["ID"]).join(df_feats, on="ID", how="left")
    df_train = preprocess.cast_dtype(df_train)
    df_train = utils.reduce_memory_usage_pl(df_train, "df_train")
    logger.info(f"{df_train = }")
    logger.info(f"{df_train.columns = }")

    drop_cols = [*target_cols, "ID", "fold", "scene_id", "scene_time", "index", "sample_id"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols + target_cols]
    category_cols = [c for c in feature_cols if df_train[c].dtype == pl.Categorical]

    utils.pinfo(feature_cols)

    print(f"{df_train = }")
    print(f"{oof_nn = }")
    print(f"{oof_gbdt = }")

    def objective(trial: optuna.Trial) -> float:
        seed = cfg.seed
        max_depth = trial.suggest_int("max_depth", 3, 12)
        params = {
            # --- Constants Parameters
            "learning_rate": 0.005,
            "verbosity": -1,
            "seed": seed,
            "boosting_type": "gbdt",
            "n_jobs": -1,
            "drop_seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "metric": "mae",
            "objective": "regression_l1",
            # --- Tuned Parameters
            "max_depth": max_depth,
            "num_leaves": int(0.7 * (2**max_depth)),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.001, 0.99),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.001, 0.99),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 0.99),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 0.99),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 0.99),
        }
        oof_total = pl.DataFrame()
        scores = []
        for fold in range(cfg.n_folds):
            logger.info(f"Start training fold {fold}")
            df_train_fold = df_train.filter(pl.col("fold") == fold)
            df_valid_fold = df_train.filter(pl.col("fold") != fold)
            y_preds_fold = np.zeros((len(df_valid_fold), len(target_cols)))
            for i_target, target_col in enumerate(target_cols):
                ds_train = lgb.Dataset(
                    df_train_fold.select(feature_cols).to_pandas(),
                    label=df_train_fold[target_col].to_numpy(),
                    free_raw_data=False,
                    feature_name=feature_cols,
                    categorical_feature=category_cols,
                )
                ds_valid = lgb.Dataset(
                    df_valid_fold.select(feature_cols).to_pandas(),
                    label=df_valid_fold[target_col].to_numpy(),
                    reference=ds_train,
                    free_raw_data=False,
                    feature_name=feature_cols,
                    categorical_feature=category_cols,
                )
                logger.info(f"Start training {target_col} / Fold: {fold}")
                model = lgb.train(
                    params,
                    ds_train,
                    num_boost_round=NUM_BOOST_ROUND,
                    valid_sets=[ds_train, ds_valid],
                    valid_names=["train", "valid"],
                    callbacks=[
                        lgb.log_evaluation(100),
                    ],
                )
                print(f"{ds_valid.data.columns = }")
                print(f"{ds_valid.data.dtypes = }")
                y_pred = model.predict(ds_valid.data)
                y_preds_fold[:, i_target] = y_pred

            oof = pl.concat(
                [
                    df_valid_fold,
                    pl.DataFrame({
                        **{f"pp-pred-{c}": y_preds_fold[:, i] for i, c in enumerate(target_cols)},
                    }),
                ],
                how="horizontal",
            )
            oof_total = pl.concat([oof_total, oof])

            score_fold = metrics.score(
                y_true=oof.select(target_cols).to_numpy(),
                y_pred=oof.select([f"pp-pred-{c}" for c in target_cols]).to_numpy(),
            )
            logger.info(f"Score: {score_fold}")
            scores.append(score_fold)
        score_total = metrics.score(
            y_true=oof_total.select(target_cols).to_numpy(),
            y_pred=oof_total.select([f"pp-pred-{c}" for c in target_cols]).to_numpy(),
        )
        logger.info(f"""
        ===============================================
        Exp: {cfg.name} PostProcess GBDT

        Total Score: {score_total}
        Scores: {scores}
        Mean: {np.mean(scores)} +/- {np.std(scores)}
        ===============================================
        """)
        return score_total

    study = optuna.create_study(study_name=cfg.name, direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    utils.pinfo(f"Best trial: {study.best_trial}")
    utils.pinfo(f"Best params: {study.best_params}")
    oof_total = pl.DataFrame()
    seed = cfg.seed
    params = {
        # --- Constants Parameters
        "learning_rate": 0.005,
        "verbosity": -1,
        "seed": seed,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "drop_seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "metric": "mae",
        "objective": "regression_l1",
        # --- Tuned Parameters
        **study.best_params,
    }
    scores = []
    for fold in range(cfg.n_folds):
        logger.info(f"Start training fold {fold}")
        df_train_fold = df_train.filter(pl.col("fold") == fold)
        df_valid_fold = df_train.filter(pl.col("fold") != fold)
        y_preds_fold = np.zeros((len(df_valid_fold), len(target_cols)))
        for i_target, target_col in enumerate(target_cols):
            ds_train = lgb.Dataset(
                df_train_fold.select(feature_cols).to_pandas(),
                label=df_train_fold[target_col].to_numpy(),
                free_raw_data=False,
                feature_name=feature_cols,
                categorical_feature=category_cols,
            )
            ds_valid = lgb.Dataset(
                df_valid_fold.select(feature_cols).to_pandas(),
                label=df_valid_fold[target_col].to_numpy(),
                reference=ds_train,
                free_raw_data=False,
                feature_name=feature_cols,
                categorical_feature=category_cols,
            )
            logger.info(f"Start training {target_col} / Fold: {fold}")
            model = lgb.train(
                params,
                ds_train,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[ds_train, ds_valid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.log_evaluation(100),
                ],
            )
            y_pred = model.predict(ds_valid.data)
            y_preds_fold[:, i_target] = y_pred
            model.save_model(save_dir / f"pp_gbdt_model_{fold}_{target_col}.ubj")

        oof = pl.concat(
            [
                df_valid_fold,
                pl.DataFrame({
                    **{f"pp-pred-{c}": y_preds_fold[:, i] for i, c in enumerate(target_cols)},
                }),
            ],
            how="horizontal",
        )
        oof_total = pl.concat([oof_total, oof])

        score_fold = metrics.score(
            y_true=oof.select(target_cols).to_numpy(),
            y_pred=oof.select([f"pp-pred-{c}" for c in target_cols]).to_numpy(),
        )
        logger.info(f"Score: {score_fold}")
        scores.append(score_fold)

    score_total = metrics.score(
        y_true=oof_total.select(target_cols).to_numpy(),
        y_pred=oof_total.select([f"pp-pred-{c}" for c in target_cols]).to_numpy(),
    )
    logger.info(f"""
===============================================
Exp: {cfg.name} PostProcess GBDT

Total Score: {score_total}
Scores: {scores}
Mean: {np.mean(scores)} +/- {np.std(scores)}

DURATION: {log.calc_duration_from(CALLED_TIME)}
COMMIT_HASH: {utils.get_commit_hash_head()}
===============================================
    """)


if __name__ == "__main__":
    main()
