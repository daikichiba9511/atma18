import pathlib
from typing import Mapping

import polars as pl


def fill_null(df: pl.DataFrame, fill_num_value: float = -9999, fill_str_value: str = "missing") -> pl.DataFrame:
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            df = df.with_columns(pl.col(col).fill_null(fill_num_value))
        elif dtype in [pl.Utf8, pl.String, pl.Categorical]:
            df = df.with_columns(pl.col(col).fill_null(fill_str_value))
    return df


def load_df_traffic_light(json_dir: pathlib.Path) -> pl.DataFrame:
    traffic_lights = []
    for json_fp in json_dir.glob("*.json"):
        df = pl.read_json(json_fp)
        if df.is_empty():
            continue
        traffic_lights.append(df.with_columns(ID=pl.lit(json_fp.stem)))
    if not traffic_lights:
        raise ValueError(f"No json files in {json_dir}. Maybe 'traffic_lights'?")
    df_traffic_light = pl.concat(traffic_lights, how="vertical")
    bbox_df = df_traffic_light["bbox"].list.to_struct(fields=[f"bbox_{i}" for i in range(4)]).struct.unnest()
    df_traffic_light = pl.concat([df_traffic_light, bbox_df], how="horizontal").select(pl.all().exclude("bbox"))
    return df_traffic_light


def make_df_traffic_light_count(df_traffic_light: pl.DataFrame) -> pl.DataFrame:
    return df_traffic_light.pivot(values="class", index="ID", on="class", aggregate_function="len").fill_null(0)


def add_feature(
    df: pl.DataFrame,
    agg_cols: tuple[str, ...] = ("vEgo", "aEgo", "steeringAngleDeg", "steeringTorque", "gas"),
) -> pl.DataFrame:
    df = (
        df.with_columns(
            scene_id=pl.col("ID").str.split("_").list[0],
            scene_time=pl.col("ID").str.split("_").list[1].cast(pl.Int32),
            gearShifter=pl.col("gearShifter").cast(pl.Categorical),
        )
        .sort(["scene_id", "scene_time"])
        .with_columns(*[
            *[pl.col(c).shift(-1).over("scene_id").alias(f"{c}_shift-1") for c in agg_cols],
            *[pl.col(c).shift(1).over("scene_id").alias(f"{c}_shift+1") for c in agg_cols],
            *[pl.col(c).diff(-1).over("scene_id").alias(f"{c}_diff-1") for c in agg_cols],
            *[pl.col(c).diff(1).over("scene_id").alias(f"{c}_diff+1") for c in agg_cols],
            *[pl.mean(c).over("scene_id").alias(f"{c}_mean") for c in agg_cols],
            *[pl.std(c).over("scene_id").alias(f"{c}_std") for c in agg_cols],
            *[pl.min(c).over("scene_id").alias(f"{c}_min") for c in agg_cols],
            *[pl.max(c).over("scene_id").alias(f"{c}_max") for c in agg_cols],
        ])
    )
    df_origin = df.clone()
    original_cols = df_origin.columns

    scene_stats = df.group_by("scene_id").agg(**{
        **{f"{c}_mean": pl.mean(c) for c in agg_cols},
        **{f"{c}_std": pl.std(c) for c in agg_cols},
        **{f"{c}_min": pl.min(c) for c in agg_cols},
        **{f"{c}_max": pl.max(c) for c in agg_cols},
        **{f"{c}_median": pl.median(c) for c in agg_cols},
    })

    df = df.with_columns(
        # -- 運転スタイルに関する特徴量
        harsh_acceleration=(pl.col("aEgo") > 2).cast(pl.Int32),
        harsh_braking=(pl.col("aEgo") < -2).cast(pl.Int32),
        # -- 急ハンドルの回数
        sharp_steering=(pl.col("steeringAngleDeg").abs() > 45).cast(pl.Int32),
    )
    # -- シーンごと集計
    driving_style = df.group_by("scene_id").agg(
        harsh_acceleration_sum=pl.sum("harsh_acceleration"),
        harsh_braking_sum=pl.sum("harsh_braking"),
        sharp_steering_sum=pl.sum("sharp_steering"),
    )
    # -- 運転の滑らかさに関する特徴量
    df = df.with_columns(
        speed_change_rate=df.group_by("scene_id").df["vEgo"].diff(),
        steering_change_rate=df.group_by("scene_id").df["steeringAngleDeg"].diff(),
    )
    # -- 変化率の統計量
    smoothness_stats = df.group_by("scene_id").agg(
        speed_change_rate_mean=pl.mean("speed_change_rate"),
        speed_change_rate_std=pl.std("speed_change_rate"),
        speed_change_rate_min=pl.min("speed_change_rate"),
        steering_change_rate_mean=pl.mean("steering_change_rate"),
        steering_change_rate_std=pl.std("steering_change_rate"),
        steering_change_rate_min=pl.min("steering_change_rate"),
    )

    # -- ペダルの操作量に関する特徴量
    pedal_features = df.group_by("scene_id").agg(
        brakePressed_sum=pl.sum("brakePressed"),
        gasPressed_sum=pl.sum("gasPressed"),
        brake_mean=pl.mean("brake"),
    )

    # -- ウィンカーの操作量に関する特徴量
    blinker_features = df.group_by("scene_id").agg(
        leftBlinker_sum=pl.sum("leftBlinker"),
        rightBlinker_sum=pl.sum("rightBlinker"),
    )

    # -- 高度な特徴量
    df = df.with_columns(
        speed_zone=pl.col("vEgo").qcut(5, labels=["very_slow", "slow", "normal", "fast", "very_fast"]),
        acc_zone=pl.col("aEgo").qcut(
            5, labels=["very_slow_acc", "slow_acc", "normal_acc", "fast_acc", "very_fast_acc"]
        ),
    )
    speed_profile = (
        df.select(["scene_id", "speed_zone"])
        .pivot(index="scene_id", on="speed_zone", values="speed_zone", aggregate_function="len")
        .fill_null(0)
    )
    acc_profile = (
        df.select(["scene_id", "acc_zone"])
        .pivot(index="scene_id", on="acc_zone", values="acc_zone", aggregate_function="len")
        .fill_null(0)
    )

    all_features = (
        scene_stats.join(
            driving_style,
            on="scene_id",
            how="left",
        )
        .join(
            smoothness_stats,
            on="scene_id",
            how="left",
        )
        .join(
            pedal_features,
            on="scene_id",
            how="left",
        )
        .join(
            blinker_features,
            on="scene_id",
            how="left",
        )
        .join(
            speed_profile,
            on="scene_id",
            how="left",
        )
        .join(
            acc_profile,
            on="scene_id",
            how="left",
        )
    )
    # drop_cols = sorted(list(set(original_cols) & set(all_features.columns)))
    # all_features = all_features.drop(drop_cols)
    df = df_origin.join(all_features.fill_null(0), on="scene_id", how="left")

    return df


if __name__ == "__main__":
    from src import constants

    df = pl.read_csv(constants.DATA_DIR / "train_features.csv")
    df = add_feature(df)
    print(df)
