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
    return df
