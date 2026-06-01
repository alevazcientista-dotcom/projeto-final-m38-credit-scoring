"""Funções de pré-processamento, treino e escoragem para o Projeto Final M38.

Este módulo foi pensado para ser usado tanto no notebook quanto no app Streamlit.
O pickle do modelo depende deste arquivo, então mantenha-o no mesmo repositório.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "mau"
DATE_COL = "data_ref"
ID_COL = "index"

CATEGORICAL_FEATURES = [
    "sexo",
    "posse_de_veiculo",
    "posse_de_imovel",
    "tipo_renda",
    "educacao",
    "estado_civil",
    "tipo_residencia",
]

NUMERIC_FEATURES = [
    "qtd_filhos",
    "idade",
    "tempo_emprego",
    "qt_pessoas_residencia",
    "renda",
    "renda_log",
    "tempo_emprego_missing",
    "tempo_emprego_zero",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + [
    "qtd_filhos",
    "idade",
    "tempo_emprego",
    "qt_pessoas_residencia",
    "renda",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria variáveis simples e robustas antes do ColumnTransformer.

    - renda_log: transformação logarítmica da renda.
    - tempo_emprego_missing: indicador de ausência em tempo_emprego.
    - tempo_emprego_zero: indicador para zero estrutural ou valor não positivo.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in MODEL_FEATURES:
            if col not in X.columns:
                X[col] = np.nan

        X["renda"] = pd.to_numeric(X["renda"], errors="coerce")
        X["tempo_emprego"] = pd.to_numeric(X["tempo_emprego"], errors="coerce")

        X["renda_log"] = np.log1p(X["renda"].clip(lower=0))
        X["tempo_emprego_missing"] = X["tempo_emprego"].isna().astype(int)
        X["tempo_emprego_zero"] = (X["tempo_emprego"].fillna(0) <= 0).astype(int)

        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("object")

        return X[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita outliers pelos quantis aprendidos no treino."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibilidade entre versões do sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model_pipeline() -> Pipeline:
    """Pipeline completo de pré-processamento + regressão logística."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=None,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessamento", preprocessor),
            ("modelo", model),
        ]
    )


def load_credit_data(path: str) -> pd.DataFrame:
    """Carrega .ftr/.feather ou .csv."""
    if path.lower().endswith((".ftr", ".feather")):
        return pd.read_feather(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Formato não suportado. Use .ftr, .feather ou .csv")


def split_dev_oot(df: pd.DataFrame, n_oot_months: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os últimos meses de data_ref como OOT."""
    base = df.copy()
    base[DATE_COL] = pd.to_datetime(base[DATE_COL])
    months = sorted(base[DATE_COL].dt.to_period("M").unique())
    oot_months = months[-n_oot_months:]
    is_oot = base[DATE_COL].dt.to_period("M").isin(oot_months)
    return base.loc[~is_oot].copy(), base.loc[is_oot].copy()


def prepare_xy(df: pd.DataFrame):
    """Retorna X e y removendo alvo, data e identificador."""
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    drop_cols = [c for c in [TARGET, DATE_COL, ID_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def sample_training_data(df: pd.DataFrame, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Amostra opcional estratificada pelo alvo para treino mais rápido."""
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df.copy()
    frac = sample_size / len(df)
    return (
        df.groupby(TARGET, group_keys=False)
        .apply(lambda x: x.sample(max(1, int(len(x) * frac)), random_state=random_state))
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def predict_proba_mau(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X, _ = prepare_xy(df)
    return model.predict_proba(X)[:, 1]


def calculate_metrics(y_true, y_score, cutoff: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= cutoff).astype(int)

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float(np.max(tpr - fpr))
    gini = float(2 * auc - 1)
    acc = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Acurácia": acc,
        "AUC": float(auc),
        "Gini": gini,
        "KS": ks,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def score_dataframe(model: Pipeline, df: pd.DataFrame, cutoff: float = 0.5) -> pd.DataFrame:
    """Adiciona score e classificação prevista a uma base."""
    scored = df.copy()
    scored["score_mau"] = predict_proba_mau(model, scored)
    scored["classe_prevista"] = np.where(scored["score_mau"] >= cutoff, "mau", "bom")
    return scored
