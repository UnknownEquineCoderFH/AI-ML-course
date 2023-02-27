from __future__ import annotations

import numpy as np
import polars as pl
from sklearn import model_selection


def make_dataframe() -> pl.DataFrame:
    classes = [*("A" * 800), *("B" * 150), *("C" * 50)]
    np.random.shuffle(classes)

    data = {
        "col1": np.random.randint(1, 11, 1000),
        "col2": classes,
    }

    df = pl.DataFrame(data)

    return df


def manual_split(
    df: pl.DataFrame, outfile: str = "out/csv/manual"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    size = df.shape[0]
    train_size, test_size = int(size * 0.9), int(size * 0.1)

    train_df = df.slice(0, train_size)
    test_df = df.slice(train_size, test_size)

    train_df.write_csv(outfile + "_train.csv", has_header=False)
    test_df.write_csv(outfile + "_test.csv", has_header=False)

    return train_df, test_df


def sklearn_split(
    df: pl.DataFrame, outfile: str = "out/csv/sklearn"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_df, test_df = model_selection.train_test_split(
        df, test_size=0.1, stratify=df["col2"]
    )

    train_df.write_csv(outfile + "_train.csv", has_header=False)  # type: ignore
    test_df.write_csv(outfile + "_test.csv", has_header=False)  # type: ignore

    return train_df, test_df  # type: ignore


def main() -> int:
    df = make_dataframe()
    train, test = manual_split(df)

    sktrain, sktest = sklearn_split(df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
