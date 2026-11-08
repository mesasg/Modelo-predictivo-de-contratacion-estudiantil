"""Métricas de evaluación: ROC-AUC (principal), F1, recall y precisión de la clase 0, accuracy."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluar(modelo, X, y):
    """Evalúa un modelo ajustado sobre datos que no se usaron para ajustarlo."""
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1_clase0": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "recall_clase0": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        # Proporción de alertas de "no contratado" que son correctas
        "precision_clase0": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        # Filas: clase real (0, 1); columnas: clase predicha (0, 1)
        "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def validacion_cruzada(modelo, X, y, folds, semilla):
    """Validación cruzada estratificada solo sobre el entrenamiento.

    cross_validate reajusta el pipeline completo en cada partición, así que el
    preprocesamiento nunca ve los datos con los que se evalúa.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=semilla)
    scoring = {
        "roc_auc": "roc_auc",
        "f1_clase0": make_scorer(f1_score, pos_label=0, zero_division=0),
        "recall_clase0": make_scorer(recall_score, pos_label=0, zero_division=0),
        "precision_clase0": make_scorer(precision_score, pos_label=0, zero_division=0),
        "accuracy": "accuracy",
    }
    res = cross_validate(modelo, X, y, cv=cv, scoring=scoring)
    resumen = {"folds": folds}
    for nombre in scoring:
        valores = res[f"test_{nombre}"]
        resumen[nombre] = {"media": float(valores.mean()), "desviacion": float(valores.std())}
    return resumen
