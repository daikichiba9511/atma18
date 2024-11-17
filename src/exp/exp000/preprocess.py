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


class Preprocessor:
    df: pl.DataFrame

    def __init__(self) -> None:
        pass

    @classmethod
    def load(cls, fp_dict: Mapping[str, pathlib.Path]) -> "Preprocessor":
        self = cls()
        return self

    def fit(self, df: pl.DataFrame) -> "Preprocessor":
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            scene_id=pl.col("ID").str.split("_").list[0],
            scene_time=pl.col("ID").str.split("_").list[1].cast(pl.Int32),
            gearShifter=pl.col("gearShifter").cast(pl.Categorical),
        )
        return df

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)
