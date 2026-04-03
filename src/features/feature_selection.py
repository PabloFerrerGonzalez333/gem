"""
Utilidades de selección de características para clasificación binaria.

Incluye tres estrategias principales:

1. :func:`sel_varianza`: Filtra columnas con baja varianza.
2. :func:`sel_correlacion`: Selecciona por correlación con la variable objetivo.
3. :func:`sel_colinealidad`: Detecta pares muy correlacionados y sugiere cuál
   eliminar según su relación con el objetivo.
4. :func:`statistical_select`: Orquesta los tres pasos anteriores.
5. :func:`rfr_select`: Selección vía RFECV con ``RandomForestClassifier``.

Notas
-----
- Se asume que ``X`` y ``y`` son numéricos (sin categorías sin codificar).
- Para ``rfr_select`` se espera que las matrices ya estén escaladas/encoded.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, VarianceThreshold

RANDOM_STATE = 42
SEED = 42
np.random.seed(SEED)


def sel_varianza(X_train: pd.DataFrame) -> List[str]:
    """
    Elimina (lógicamente) características con baja varianza en ``X_train``.

    Aplica un umbral de varianza de ``0.001`` y devuelve los nombres de las
    columnas que **no** superan el umbral (es decir, candidatas a eliminar).
    No modifica ``X_train`` in-place.

    :param X_train: Matriz de entrenamiento (numérica).
    :type X_train: pandas.DataFrame
    :return: Lista de columnas con varianza < 0.001.
    :rtype: list[str]

    **Ejemplo**

    .. code-block:: python

       bajas = sel_varianza(X_train)
       X_filtrado = X_train.drop(columns=bajas)
    """
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train)

    low_variance_columns = X_train.columns[~selector.get_support()]

    if len(low_variance_columns) > 0:
        print("Columnas con baja varianza (se eliminarán):")
        print(low_variance_columns.tolist())
    else:
        print("No hay columnas con baja varianza.")

    return low_variance_columns.tolist()


def sel_correlacion(
    X_sel: pd.DataFrame,
    y_sel: pd.Series | np.ndarray,
    thres: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Selecciona características por correlación absoluta con ``y``.

    Calcula la correlación de Pearson de cada columna de ``X_sel`` con la
    variable objetivo, construye un DataFrame ordenado por ``|corr|`` y
    devuelve la lista de columnas con ``|corr| > thres``.

    :param X_sel: Matriz de atributos (numérica).
    :type X_sel: pandas.DataFrame
    :param y_sel: Vector objetivo (1D).
    :type y_sel: pandas.Series | numpy.ndarray
    :param thres: Umbral de selección por ``|corr|`` (p. ej., ``0.02``).
    :type thres: float
    :return:
        - **corrs** (*pandas.DataFrame*): Tabla con columna ``vals`` (``|corr|``)
          ordenada descendentemente e indexada por nombre de feature.
        - **selected_features** (*list[str]*): Columnas con ``|corr| > thres``.
    :rtype: tuple[pandas.DataFrame, list[str]]

    **Notas**
    -------
    Requiere columnas numéricas; si hay NaN, la correlación puede ser ``NaN``.
    """
    if isinstance(y_sel, pd.DataFrame):
        y_vec = y_sel.iloc[:, 0]
    else:
        y_vec = pd.Series(y_sel) if not isinstance(y_sel, pd.Series) else y_sel

    correlations = X_sel.apply(lambda x: x.corr(y_vec))
    corrs = pd.DataFrame(
        {"vals": correlations.abs().values}, index=correlations.index
    ).sort_values("vals", ascending=False)
    selected_features = corrs.index[corrs["vals"] > thres].tolist()
    return corrs, selected_features


def sel_colinealidad(
    X_train: pd.DataFrame,
    thres: float,
    y_train: pd.Series | np.ndarray,
    v_corr: float,
) -> List[str]:
    """
    Detecta pares de características con alta colinealidad y sugiere cuál
    eliminar según su correlación con ``y``.

    Pasos:
      1. Calcula la matriz de correlación absoluta de ``X_train``.
      2. Busca pares con correlación > ``thres`` (solo triángulo superior).
      3. Para cada par, compara ``|corr(feature, y)|`` y sugiere eliminar la
         de menor valor absoluto.

    :param X_train: Matriz de entrenamiento (numérica).
    :type X_train: pandas.DataFrame
    :param thres: Umbral de colinealidad entre *features* (p. ej., ``0.9``).
    :type thres: float
    :param y_train: Vector objetivo (1D).
    :type y_train: pandas.Series | numpy.ndarray
    :param v_corr: Umbral para calcular/filtrar correlaciones con ``y`` al
                   priorizar qué columna descartar (se usa internamente).
    :type v_corr: float
    :return: Lista de columnas sugeridas para eliminar por colinealidad.
    :rtype: list[str]
    """
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [
        (row, column)
        for row in upper.index
        for column in upper.columns
        if upper.loc[row, column] > thres
    ]

    features_to_drop: set[str] = set()

    corrs, _ = sel_correlacion(X_train, y_train, v_corr)
    target_correlations = corrs["vals"]

    for feature1, feature2 in high_corr_pairs:
        corr1 = target_correlations.get(feature1, 0.0)
        corr2 = target_correlations.get(feature2, 0.0)

        feature_to_drop = feature2 if abs(corr1) > abs(corr2) else feature1
        features_to_drop.add(feature_to_drop)

        print(
            "Pares altamente correlacionados: "
            f"{feature1} y {feature2} (corr: {corr_matrix.loc[feature1, feature2]:.2f})"
        )
        print(
            f"Correlación con objetivo - {feature1}: {corr1:.2f}, "
            f"{feature2}: {corr2:.2f}"
        )
        print(f"Sugerencia de eliminar: {feature_to_drop}\n")

    return list(features_to_drop)


def statistical_select(
    X_train_transformed: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test_transformed: pd.DataFrame,
    v_corr: float,
    v_col: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selección estadística por varianza, correlación y colinealidad.

    Pipeline:
      1. Filtra columnas con baja varianza (``sel_varianza``).
      2. Selecciona columnas con ``|corr| > v_corr`` respecto a ``y``.
      3. De las anteriores, elimina las con pares muy correlacionados
         (``> v_col``), conservando la más relacionada con ``y``.
      4. Aplica el mismo subconjunto a ``X_test_transformed``.

    :param X_train_transformed: Entrenamiento (numérico, ya transformado).
    :type X_train_transformed: pandas.DataFrame
    :param y_train: Vector objetivo (1D).
    :type y_train: pandas.Series | numpy.ndarray
    :param X_test_transformed: Test con mismas columnas que ``X_train_transformed``.
    :type X_test_transformed: pandas.DataFrame
    :param v_corr: Umbral de ``|corr|`` con ``y`` (p. ej., ``0.02``).
    :type v_corr: float
    :param v_col: Umbral de colinealidad entre *features* (p. ej., ``0.9``).
    :type v_col: float
    :return: ``(X_train_selected, X_test_selected)`` con columnas filtradas.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    cols_low_var = sel_varianza(X_train_transformed)

    _, cols_high_corr = sel_correlacion(X_train_transformed, y_train, v_corr)
    cols = list(set(cols_high_corr) - set(cols_low_var))

    cols_high_col = sel_colinealidad(X_train_transformed[cols], v_col, y_train, v_corr)
    cols = list(set(cols) - set(cols_high_col))

    X_train_selected = X_train_transformed[cols]
    X_test_selected = X_test_transformed[cols]

    print(
        "Statistical - Número de características seleccionadas: "
        f"{X_train_selected.shape[1]}, inicialmente {X_train_transformed.shape[1]}"
    )

    return X_train_selected, X_test_selected


def rfr_select(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test_scaled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selección de características con RFECV + RandomForestClassifier.

    Usa Eliminación Recursiva de Características con Validación Cruzada
    (RFECV) y un ``RandomForestClassifier`` para quedarse con las columnas
    más relevantes según la métrica ``roc_auc``.

    :param X_train_scaled: Entrenamiento ya escalado/encodeado.
    :type X_train_scaled: pandas.DataFrame
    :param y_train: Vector objetivo (1D). Se convierte internamente a 1D.
    :type y_train: pandas.Series | numpy.ndarray
    :param X_test_scaled: Conjunto de test con mismas columnas que ``X_train_scaled``.
    :type X_test_scaled: pandas.DataFrame
    :return: ``(X_train_selected, X_test_selected)`` con columnas seleccionadas.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]

    **Notas**
    ------
    - ``scoring='roc_auc'`` asume un problema binario con etiquetas {0, 1}.
    - Ajusta ``n_estimators``/``cv`` según tamaño de muestra y coste.
    """
    rf_estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=200,
    )

    rfecv = RFECV(
        estimator=rf_estimator,
        step=1,
        cv=5,
        scoring="roc_auc",
        min_features_to_select=1,
        n_jobs=-1,
    )

    y_1d = np.ravel(y_train)
    rfecv.fit(X_train_scaled, y_1d)

    idx = np.where(rfecv.support_)[0]
    names = X_train_scaled.columns[idx]

    X_train_selected = pd.DataFrame(X_train_scaled.iloc[:, idx], columns=names)
    X_test_selected = pd.DataFrame(X_test_scaled.iloc[:, idx], columns=names)

    print(
        "Forest - Número de características seleccionadas: "
        f"{X_train_selected.shape[1]}, originalmente {X_train_scaled.shape[1]}"
    )

    return X_train_selected, X_test_selected
