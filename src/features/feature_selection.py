"""
Feature selection utilities for binary classification.

Includes three main strategies:

1. :func:`sel_varianza`: Filters columns with low variance.
2. :func:`sel_correlacion`: Selects by correlation with the target variable.
3. :func:`sel_colinealidad`: Detects highly correlated pairs and suggests which
   one to eliminate based on its relationship with the target.
4. :func:`statistical_select`: Orchestrates the three previous steps.
5. :func:`rfr_select`: Selection via RFECV with ``RandomForestClassifier``.

Notes
-----
- It is assumed that ``X`` and ``y`` are numerical (no unencoded categories).
- For ``rfr_select`` it is expected that the matrices are already scaled/encoded.
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
    Logically removes features with low variance in ``X_train``.

    Applies a variance threshold of ``0.001`` and returns the names of the
    columns that do **not** exceed the threshold (i.e., candidates for removal).
    Does not modify ``X_train`` in-place.

    :param X_train: Training matrix (numerical).
    :type X_train: pandas.DataFrame
    :return: List of columns with variance < 0.001.
    :rtype: list[str]

    **Example**

    .. code-block:: python

       low_var = sel_varianza(X_train)
       X_filtered = X_train.drop(columns=low_var)
    """
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train)

    low_variance_columns = X_train.columns[~selector.get_support()]

    if len(low_variance_columns) > 0:
        print("Columns with low variance (to be removed):")
        print(low_variance_columns.tolist())
    else:
        print("No columns with low variance found.")

    return low_variance_columns.tolist()


def sel_correlacion(
    X_sel: pd.DataFrame,
    y_sel: pd.Series | np.ndarray,
    thres: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Selects features by absolute correlation with ``y``.

    Calculates the Pearson correlation of each column in ``X_sel`` with the
    target variable, builds a DataFrame sorted by ``|corr|`` and
    returns the list of columns with ``|corr| > thres``.

    :param X_sel: Feature matrix (numerical).
    :type X_sel: pandas.DataFrame
    :param y_sel: Target vector (1D).
    :type y_sel: pandas.Series | numpy.ndarray
    :param thres: Selection threshold for ``|corr|`` (e.g., ``0.02``).
    :type thres: float
    :return:
        - **corrs** (*pandas.DataFrame*): Table with column ``vals`` (``|corr|``)
          sorted descending and indexed by feature name.
        - **selected_features** (*list[str]*): Columns with ``|corr| > thres``.
    :rtype: tuple[pandas.DataFrame, list[str]]

    **Notes**
    -------
    Requires numerical columns; if there are NaNs, the correlation may be ``NaN``.
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
    Detects feature pairs with high collinearity and suggests which to
    eliminate based on its correlation with ``y``.

    Steps:
      1. Computes the absolute correlation matrix of ``X_train``.
      2. Finds pairs with correlation > ``thres`` (upper triangle only).
      3. For each pair, compares ``|corr(feature, y)|`` and suggests removing the
         one with the lower absolute value.

    :param X_train: Training matrix (numerical).
    :type X_train: pandas.DataFrame
    :param thres: Collinearity threshold between *features* (e.g., ``0.9``).
    :type thres: float
    :param y_train: Target vector (1D).
    :type y_train: pandas.Series | numpy.ndarray
    :param v_corr: Threshold to compute/filter correlations with ``y`` when
                   prioritizing which column to drop (used internally).
    :type v_corr: float
    :return: List of suggested columns to remove due to collinearity.
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
            "Highly correlated pairs: "
            f"{feature1} and {feature2} (corr: {corr_matrix.loc[feature1, feature2]:.2f})"
        )
        print(
            f"Correlation with target - {feature1}: {corr1:.2f}, "
            f"{feature2}: {corr2:.2f}"
        )
        print(f"Suggestion to drop: {feature_to_drop}\n")

    return list(features_to_drop)


def statistical_select(
    X_train_transformed: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test_transformed: pd.DataFrame,
    v_corr: float,
    v_col: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Statistical selection by variance, correlation, and collinearity.

    Pipeline:
      1. Filters out low variance columns (``sel_varianza``).
      2. Selects columns with ``|corr| > v_corr`` regarding ``y``.
      3. From the prior ones, drops those with highly correlated pairs
         (``> v_col``), keeping the one most correlated with ``y``.
      4. Applies the same subset to ``X_test_transformed``.

    :param X_train_transformed: Training set (numerical, already transformed).
    :type X_train_transformed: pandas.DataFrame
    :param y_train: Target vector (1D).
    :type y_train: pandas.Series | numpy.ndarray
    :param X_test_transformed: Test set with same columns as ``X_train_transformed``.
    :type X_test_transformed: pandas.DataFrame
    :param v_corr: Threshold of ``|corr|`` with ``y`` (e.g., ``0.02``).
    :type v_corr: float
    :param v_col: Collinearity threshold between *features* (e.g., ``0.9``).
    :type v_col: float
    :return: ``(X_train_selected, X_test_selected)`` with filtered columns.
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
        "Statistical - Number of selected features: "
        f"{X_train_selected.shape[1]}, initially {X_train_transformed.shape[1]}"
    )

    return X_train_selected, X_test_selected


def rfr_select(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test_scaled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Feature selection using RFECV + RandomForestClassifier.

    Uses Recursive Feature Elimination with Cross-Validation
    (RFECV) and a ``RandomForestClassifier`` to keep the most relevant
    columns according to the ``roc_auc`` metric.

    :param X_train_scaled: Training set already scaled/encoded.
    :type X_train_scaled: pandas.DataFrame
    :param y_train: Target vector (1D). Converted internally to 1D.
    :type y_train: pandas.Series | numpy.ndarray
    :param X_test_scaled: Test set with same columns as ``X_train_scaled``.
    :type X_test_scaled: pandas.DataFrame
    :return: ``(X_train_selected, X_test_selected)`` with selected columns.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]

    **Notes**
    ------
    - ``scoring='roc_auc'`` assumes a binary problem with {0, 1} labels.
    - Adjust ``n_estimators``/``cv`` according to sample size and cost.
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
        "Forest - Number of selected features: "
        f"{X_train_selected.shape[1]}, originally {X_train_scaled.shape[1]}"
    )

    return X_train_selected, X_test_selected
