"""
Construcción y preparación de *features* para modelado (Telco Churn).

Este módulo realiza:
  1) Imputación manual de valores.
  2) Codificación *one-hot* de categóricas.
  3) *Split* train/test + escalado robusto de numéricas.
  4) (Opcional) Balanceo con SMOTE y selección de características
     estadística + RFECV.
  5) Persistencia en Excel de los conjuntos resultantes.

Uso CLI
-------
.. code-block:: bash

   python -m src.features.build_features \
       --in data/preprocessed/telco_preprocessed.xlsx \
       --out data/processed \
       --kind cc

Donde ``kind`` es una cadena de dos caracteres:
  - 1º char: ``'c'`` aplica SMOTE (class balancing), ``'s'`` no.
  - 2º char: ``'c'`` aplica *feature engineering* (selección), ``'s'`` no.
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
#       Utilidades
# ==========================


def get_cols_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Determina columnas por tipo semántico a partir de prefijos.

    :param df: DataFrame de entrada (tras renombrado de columnas).
    :type df: pandas.DataFrame
    :return:
        - **cat_cols**: Columnas categóricas (prefijo ``c_``).
        - **num_cols**: Columnas numéricas (prefijos ``f_`` o ``i_``).
        - **bin_cols**: Columnas binarias (prefijo ``b_``).
        - **str_cols**: Columnas string/identificadores (prefijo ``s_``).
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
    Imputa valores faltantes de forma manual en columnas numéricas conocidas.

    Actualmente:
      - ``f_total``: faltantes → ``0``.

    :param df: DataFrame original.
    :type df: pandas.DataFrame
    :return: Copia imputada.
    :rtype: pandas.DataFrame
    """
    out = df.copy()
    if "f_total" in out.columns:
        out["f_total"] = out["f_total"].fillna(0)
    return out


def codificacion_categoricas(df: pd.DataFrame, cat_cols: Sequence[str]) -> pd.DataFrame:
    """
    Aplica *one-hot encoding* a columnas categóricas.

    :param df: DataFrame de entrada.
    :type df: pandas.DataFrame
    :param cat_cols: Columnas categóricas a codificar.
    :type cat_cols: Sequence[str]
    :return: DataFrame con variables dummies (sin *drop_first*).
    :rtype: pandas.DataFrame
    """
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=list(cat_cols), drop_first=False, dtype=int)


def escalado_numericas(
    df: pd.DataFrame, num_cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Realiza *split* train/test estratificado y escala numéricas con RobustScaler.

    :param df: DataFrame tras imputación y *one-hot*.
    :type df: pandas.DataFrame
    :param num_cols: Columnas numéricas a escalar.
    :type num_cols: Sequence[str]
    :return: ``(X_train, y_train, X_test, y_test)`` con escalado aplicado.
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

    # y como Series 1D (facilita uso en sklearn/imbalanced-learn)
    return X_train, y_train, X_test, y_test


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTE para balancear la clase minoritaria.

    :param X_train: Matriz de entrenamiento.
    :type X_train: pandas.DataFrame
    :param y_train: Etiquetas (Series 1D).
    :type y_train: pandas.Series
    :return: Conjunto balanceado ``(X_train_bal, y_train_bal)``.
    :rtype: tuple[pandas.DataFrame, pandas.Series]
    """
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    # Garantizar que y retorna como Series con nombre
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
    Aplica pasos opcionales (SMOTE y selección de *features*) y persiste a Excel.

    :param X_train: Entrenamiento.
    :type X_train: pandas.DataFrame
    :param y_train: Etiquetas de entrenamiento.
    :type y_train: pandas.Series
    :param X_test: Test.
    :type X_test: pandas.DataFrame
    :param y_test: Etiquetas de test.
    :type y_test: pandas.Series
    :param kind: Dos letras: ``c/s`` para SMOTE y ``c/s`` para selección.
    :type kind: str
    :param out_dir: Carpeta destino para los Excel generados.
    :type out_dir: str | pathlib.Path
    :return: Nada; escribe ficheros ``X_train_{kind}.xlsx``, etc.
    :rtype: None
    """
    if len(kind) != 2 or any(k not in {"c", "s"} for k in kind):
        raise ValueError("kind debe ser de longitud 2 con caracteres en {'c','s'}.")

    do_class_balance = kind[0] == "c"
    do_feature_eng = kind[1] == "c"

    if do_class_balance:
        X_train, y_train = apply_smote(X_train, y_train)

    if do_feature_eng:
        # Selección estadística (varianza/corr/colinealidad)
        X_train, X_test = statistical_select(
            X_train, y_train, X_test, v_corr=0.01, v_col=0.95
        )
        # RFECV con RandomForest
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
    Ejecuta el *pipeline* de *feature building* extremo a extremo.

    :param inp: Ruta al Excel preprocesado de entrada.
    :type inp: str | pathlib.Path
    :param out: Carpeta de salida para los ficheros generados.
    :type out: str | pathlib.Path
    :param kind: Dos letras: ``c/s`` para SMOTE y ``c/s`` para selección.
    :type kind: str
    :return: Nada.
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
    Construye el *argument parser* del CLI.

    :return: Parser configurado.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Limpia, codifica, escala y selecciona *features* para modelado."
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default="data/preprocessed/telco_preprocessed.xlsx",
        help="Ruta del Excel preprocesado de entrada.",
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        default="data/processed",
        help="Carpeta de salida para los Excel generados.",
    )
    parser.add_argument(
        "--kind",
        dest="kind",
        type=str,
        default="cc",
        help="Dos letras: 1) c/s (SMOTE on/off), 2) c/s (selección on/off).",
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
