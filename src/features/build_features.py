"""
Build and preparation of *features* for modeling (Telco Churn).

This module performs:
  1) Manual imputation of values.
  2) *One-hot* encoding of categoricals.
  3) Train/test *split* + robust scaling of numerics.
  4) (Optional) Balancing with SMOTE and statistical feature selection
     + RFECV.
  5) Persistence of resulting datasets to Excel.

CLI Usage
-------
.. code-block:: bash

   python -m src.features.build_features \
       --in data/preprocessed/telco_preprocessed.xlsx \
       --out data/processed \
       --kind cc

Where ``kind`` is a two-character string:
  - 1st char: ``'c'`` applies SMOTE (class balancing), ``'s'`` does not.
  - 2nd char: ``'c'`` applies *feature engineering* (selection), ``'s'`` does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.features.feature_selection import rfr_select, statistical_select

RANDOM_STATE = 42
SEED = 42
TARGET = "b_churn"


# ==========================
#       Utilities
# ==========================


def get_cols_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Determines columns by semantic type based on prefixes.

    :param df: Input DataFrame (after column renaming).
    :type df: pandas.DataFrame
    :return:
        - **cat_cols**: Categorical columns (prefix ``c_``).
        - **num_cols**: Numerical columns (prefixes ``f_`` or ``i_``).
        - **bin_cols**: Binary columns (prefix ``b_``).
        - **str_cols**: String/identifier columns (prefix ``s_``).
    :rtype: tuple[list[str], list[str], list[str], list[str]]
    """
    cat_cols = [c for c in df.columns if c.startswith("c_") and c != TARGET]
    num_cols = [
        c
        for c in df.columns
        if (c.startswith("f_") or c.startswith("i_")) and c != TARGET
    ]
    bin_cols = [c for c in df.columns if c.startswith("b_") and c != TARGET]
    str_cols = [c for c in df.columns if c.startswith("s_") and c != TARGET]
    return cat_cols, num_cols, bin_cols, str_cols


def imputacion_manual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manually imputes missing values in known numerical columns.

    Currently:
      - ``f_total``: missing → ``0``.

    :param df: Original DataFrame.
    :type df: pandas.DataFrame
    :return: Imputed copy.
    :rtype: pandas.DataFrame
    """
    out = df.copy()
    if "f_total" in out.columns:
        out["f_total"] = out["f_total"].fillna(0)
    return out


def codificacion_categoricas(df: pd.DataFrame, cat_cols: Sequence[str]) -> pd.DataFrame:
    """
    Applies *one-hot encoding* to categorical columns.

    :param df: Input DataFrame.
    :type df: pandas.DataFrame
    :param cat_cols: Categorical columns to encode.
    :type cat_cols: Sequence[str]
    :return: DataFrame with dummy variables (without *drop_first*).
    :rtype: pandas.DataFrame
    """
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=list(cat_cols), drop_first=False, dtype=int)


def escalado_numericas(
    df: pd.DataFrame, num_cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Performs stratified train/test *split* and scales numerics with RobustScaler.

    :param df: DataFrame after imputation and *one-hot*.
    :type df: pandas.DataFrame
    :param num_cols: Numerical columns to scale.
    :type num_cols: Sequence[str]
    :return: ``(X_train, y_train, X_test, y_test)`` with applied scaling.
    :rtype: tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series]
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    if num_cols:
        scaler = RobustScaler()
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train.loc[:, list(num_cols)] = scaler.fit_transform(
            X_train.loc[:, list(num_cols)]
        )
        X_test.loc[:, list(num_cols)] = scaler.transform(X_test.loc[:, list(num_cols)])

    # y as 1D Series (facilitates use in sklearn/imbalanced-learn)
    return X_train, y_train, X_test, y_test


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE to balance the minority class.

    :param X_train: Training matrix.
    :type X_train: pandas.DataFrame
    :param y_train: Labels (1D Series).
    :type y_train: pandas.Series
    :return: Balanced set ``(X_train_bal, y_train_bal)``.
    :rtype: tuple[pandas.DataFrame, pandas.Series]
    """
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    # Ensure y returns as Series with name
    if not isinstance(y_bal, pd.Series):
        y_bal = pd.Series(y_bal, name=y_train.name or TARGET)
    return X_bal, y_bal


def final_gen(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    kind: str,
    out_dir: str | Path,
) -> None:
    """
    Applies optional steps (SMOTE and *feature* selection) and persists to Excel.

    :param X_train: Training set.
    :type X_train: pandas.DataFrame
    :param y_train: Training labels.
    :type y_train: pandas.Series
    :param X_test: Test set.
    :type X_test: pandas.DataFrame
    :param y_test: Test labels.
    :type y_test: pandas.Series
    :param kind: Two letters: ``c/s`` for SMOTE and ``c/s`` for selection.
    :type kind: str
    :param out_dir: Destination folder for generated Excels.
    :type out_dir: str | pathlib.Path
    :return: Nothing; writes files ``X_train_{kind}.xlsx``, etc.
    :rtype: None
    """
    if len(kind) != 2 or any(k not in {"c", "s"} for k in kind):
        raise ValueError("kind must be of length 2 with characters in {'c','s'}.")

    do_class_balance = kind[0] == "c"
    do_feature_eng = kind[1] == "c"

    if do_class_balance:
        X_train, y_train = apply_smote(X_train, y_train)

    if do_feature_eng:
        # Statistical selection (variance/corr/collinearity)
        X_train, X_test = statistical_select(
            X_train, y_train, X_test, v_corr=0.01, v_col=0.95
        )
        # RFECV with RandomForest
        X_train, X_test = rfr_select(X_train, y_train, X_test)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X_train.to_excel(out_path / f"X_train_{kind}.xlsx", index=False)
    X_test.to_excel(out_path / f"X_test_{kind}.xlsx", index=False)
    y_train.to_excel(out_path / f"y_train_{kind}.xlsx", index=False)
    y_test.to_excel(out_path / f"y_test_{kind}.xlsx", index=False)


# ==========================
#           Main
# ==========================


def main(inp: str | Path, out: str | Path, kind: str) -> None:
    """
    Executes the end-to-end *feature building* pipeline.

    :param inp: Path to input preprocessed Excel.
    :type inp: str | pathlib.Path
    :param out: Output folder for generated files.
    :type out: str | pathlib.Path
    :param kind: Two letters: ``c/s`` for SMOTE and ``c/s`` for selection.
    :type kind: str
    :return: None.
    :rtype: None
    """
    df = pd.read_excel(inp)
    cat_cols, num_cols, _, _ = get_cols_types(df)

    df = imputacion_manual(df)
    df = codificacion_categoricas(df, cat_cols)

    X_train, y_train, X_test, y_test = escalado_numericas(df, num_cols)
    final_gen(X_train, y_train, X_test, y_test, kind, out)


def _build_parser() -> argparse.ArgumentParser:
    """
    Builds the CLI *argument parser*.

    :return: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Cleans, encodes, scales and selects *features* for modeling."
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default="data/preprocessed/telco_preprocessed.xlsx",
        help="Path to input preprocessed Excel.",
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        default="data/processed",
        help="Output folder for generated Excels.",
    )
    parser.add_argument(
        "--kind",
        dest="kind",
        type=str,
        default="cc",
        help="Two letters: 1) c/s (SMOTE on/off), 2) c/s (selection on/off).",
    )
    return parser


if __name__ == "__main__":
    PARSER = _build_parser()
    ARGS = PARSER.parse_args()
    main(ARGS.inp, ARGS.out, ARGS.kind)

    # poetry run python src/features/build_features.py \
    #   --in data/preprocessed/telco_preprocessed.xlsx \
    #   --out data/processed \
    #   --kind cc
