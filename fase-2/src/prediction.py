"""Carga de una versión guardada y generación de predicciones, sin reentrenar ni reajustar nada."""

import json
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src import config
from src.data import preparar_variables


class ErrorModelo(ValueError):
    """El artefacto no se puede cargar o no corresponde al esquema del proyecto."""


def cargar_version(ruta):
    """Carga el pipeline desde `modelo.joblib` o desde la carpeta de la versión.

    Devuelve (modelo, metadatos, avisos). Los metadatos son {} si la carpeta no contiene metadata.json.
    """
    ruta = Path(ruta)
    archivo = ruta / config.NOMBRE_MODELO if ruta.is_dir() else ruta
    if not archivo.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {archivo}")

    avisos = []
    with warnings.catch_warnings(record=True) as capturadas:
        warnings.simplefilter("always", InconsistentVersionWarning)
        try:
            modelo = joblib.load(archivo)
        # Cualquier excepción de joblib o pickle se reporta como ErrorModelo
        except Exception as e:
            raise ErrorModelo(
                f"No se pudo cargar el modelo '{archivo}': el archivo está dañado o no es un modelo "
                f"guardado con joblib ({type(e).__name__}: {e})"
            ) from e
    if any(issubclass(w.category, InconsistentVersionWarning) for w in capturadas):
        avisos.append("El modelo se guardó con otra versión de scikit-learn; instale la de requirements.txt.")

    columnas = list(getattr(modelo, "feature_names_in_", []))
    if not hasattr(modelo, "predict_proba") or sorted(columnas) != sorted(config.COLUMNAS_ENTRADA):
        raise ErrorModelo(
            f"'{archivo}' no es un modelo de este proyecto: se esperaba un pipeline con las "
            f"{len(config.COLUMNAS_ENTRADA)} variables de entrada."
        )

    metadatos = {}
    ruta_metadatos = archivo.parent / config.NOMBRE_METADATOS
    if ruta_metadatos.is_file():
        with open(ruta_metadatos, encoding="utf-8") as f:
            metadatos = json.load(f)
    return modelo, metadatos, avisos


def categorias_no_vistas(modelo, X):
    """Valores categóricos que no aparecieron en el entrenamiento, por columna."""
    encoder = modelo.named_steps["preprocesador"].named_transformers_["cat"]["encoder"]
    nuevas = {}
    for col, conocidas in zip(config.COLUMNAS_CATEGORICAS, encoder.categories_):
        valores = X[col].dropna()
        desconocidos = sorted(set(valores) - set(conocidas))
        if desconocidos:
            nuevas[col] = desconocidos
    return nuevas


def predecir(modelo, df, metadatos=None):
    """Genera una predicción por fila, en el mismo orden de `df`. Devuelve (resultados, avisos).

    - Columnas faltantes, tipos incompatibles o valores fuera de rango: ErrorDatos, sin generar predicciones.
    - Celdas vacías: se imputan con la mediana o la moda aprendidas en el entrenamiento.
    - Columnas adicionales: se ignoran y se avisan.
    - Categorías no vistas: se codifican como ceros (OneHotEncoder handle_unknown='ignore') y se avisan.
    """
    X, adicionales = preparar_variables(df)
    # Orden de columnas usado en el entrenamiento, independiente del orden del CSV
    X = X[list(modelo.feature_names_in_)]

    avisos = []
    if adicionales:
        avisos.append(f"Columnas ignoradas: {', '.join(adicionales)}")
    for col, valores in categorias_no_vistas(modelo, X).items():
        avisos.append(f"Categorías no vistas en el entrenamiento en '{col}': {valores} (se codifican como ceros)")
    celdas_vacias = int(X.isna().sum().sum())
    if celdas_vacias:
        avisos.append(f"{celdas_vacias} celdas vacías se imputaron con los valores aprendidos en el entrenamiento")

    etiquetas = {int(k): v for k, v in (metadatos or {}).get("etiquetas", config.ETIQUETAS).items()}
    prediccion = modelo.predict(X).astype(int)
    puntaje = modelo.predict_proba(X)[:, list(modelo.classes_).index(1)]

    resultados = pd.DataFrame({
        "fila": range(len(X)),
        "prediccion": prediccion,
        "etiqueta": [etiquetas[p] for p in prediccion],
        "puntaje_contratado": puntaje.round(6),
    })
    return resultados, avisos


def guardar_predicciones(resultados, ruta):
    """Escribe el CSV en un archivo temporal y lo renombra a la ruta final; si falla, elimina el temporal."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(prefix=f".{ruta.name}-", suffix=".tmp", dir=ruta.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as f:
            resultados.to_csv(f, index=False)
        # mkstemp crea el archivo solo para su dueño (600); igual que las versiones, debe poder leerse fuera de Docker
        os.chmod(temporal, 0o644)
        os.replace(temporal, ruta)
    except BaseException:
        Path(temporal).unlink(missing_ok=True)
        raise
    return ruta
