"""Construcción del preprocesamiento, del modelo y del modelo base."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS


def construir_preprocesador():
    """Numéricas: mediana + estandarización. Categóricas: moda + One-Hot.

    Las categorías no vistas en el entrenamiento se codifican como ceros (handle_unknown='ignore').
    """
    pipeline_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipeline_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", pipeline_numerico, COLUMNAS_NUMERICAS),
        ("cat", pipeline_categorico, COLUMNAS_CATEGORICAS),
    ])


def construir_modelo(semilla, C=1.0, max_iter=1000):
    """Pipeline completo de la Fase 1: preprocesamiento + Regresión Logística balanceada."""
    return Pipeline([
        ("preprocesador", construir_preprocesador()),
        ("modelo", LogisticRegression(
            class_weight="balanced", C=C, max_iter=max_iter, random_state=semilla
        )),
    ])


def construir_modelo_base():
    """Modelo base: predice siempre la clase más frecuente del entrenamiento."""
    return Pipeline([
        ("preprocesador", construir_preprocesador()),
        ("modelo", DummyClassifier(strategy="most_frequent")),
    ])
