"""Proceso completo de entrenamiento, reutilizable desde train.py, las pruebas y la futura API."""

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src import config
from src.data import ErrorDatos, leer_csv, preparar_objetivo, preparar_variables, simular_nulos
from src.evaluation import evaluar, validacion_cruzada
from src.pipeline import construir_modelo, construir_modelo_base
from src.storage import guardar_version, referencia_codigo, validar_version


def entrenar(
    ruta_datos,
    directorio_salida,
    version,
    target=config.TARGET,
    semilla=config.SEMILLA,
    tamano_prueba=config.TAMANO_PRUEBA,
    folds_cv=config.FOLDS_CV,
    tasa_nulos=config.TASA_NULOS_SIMULADOS,
    C=1.0,
    max_iter=1000,
):
    """Entrena, evalúa y guarda una versión del modelo. Devuelve métricas y rutas generadas."""
    inicio = datetime.now(timezone.utc)
    # Se valida antes de entrenar para no perder tiempo si la versión ya existe
    validar_version(directorio_salida, version)
    if target in config.COLUMNAS_ENTRADA:
        raise ErrorDatos(f"La variable objetivo '{target}' no puede ser también una variable de entrada.")

    # 1. Lectura y validación
    df = leer_csv(ruta_datos)
    y = preparar_objetivo(df, target)
    X, adicionales = preparar_variables(df)
    ignoradas = [c for c in adicionales if c != target and c not in config.COLUMNAS_EXCLUIDAS]

    # 2. Nulos simulados como en la Fase 1 (la máscara es independiente de y)
    X = simular_nulos(X, tasa_nulos, semilla)

    # 3. Separación antes de ajustar cualquier transformación. Cada fila es un estudiante
    #    distinto, así que no hay grupos que mantener juntos; se estratifica por la clase.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=tamano_prueba, random_state=semilla, stratify=y
        )
    except ValueError as e:
        raise ErrorDatos(f"No se pudo separar entrenamiento y prueba: {e}") from e

    # 4. Validación cruzada solo con el entrenamiento (el pipeline se reajusta en cada partición)
    cv = None
    if folds_cv:
        if y_train.value_counts().min() < folds_cv:
            raise ErrorDatos(
                f"La clase minoritaria tiene menos de {folds_cv} filas en el entrenamiento; "
                "reduzca --cv-folds o use más datos."
            )
        cv = validacion_cruzada(construir_modelo(semilla, C, max_iter), X_train, y_train, folds_cv, semilla)

    # 5. Ajuste final con todo el entrenamiento y evaluación en prueba (misma partición para ambos)
    modelo = construir_modelo(semilla, C, max_iter).fit(X_train, y_train)
    modelo_base = construir_modelo_base().fit(X_train, y_train)
    metricas = {
        "metrica_principal": config.METRICA_PRINCIPAL,
        "conjunto_evaluacion": "prueba",
        "n_prueba": int(len(y_test)),
        "modelo": evaluar(modelo, X_test, y_test),
        "modelo_base": evaluar(modelo_base, X_test, y_test),
        "validacion_cruzada": cv,
    }

    # 6. Metadatos y almacenamiento
    metadatos = _metadatos(
        version=version, inicio=inicio, ruta_datos=Path(ruta_datos), df=df, target=target,
        ignoradas=ignoradas, modelo=modelo, semilla=semilla, tamano_prueba=tamano_prueba,
        folds_cv=folds_cv, tasa_nulos=tasa_nulos, y_train=y_train, y_test=y_test,
    )
    rutas = guardar_version(directorio_salida, version, modelo, metricas, metadatos)
    return {"version": version, "metricas": metricas, "rutas": rutas, "columnas_ignoradas": ignoradas}


def _metadatos(*, version, inicio, ruta_datos, df, target, ignoradas, modelo, semilla,
               tamano_prueba, folds_cv, tasa_nulos, y_train, y_test):
    prep = modelo.named_steps["preprocesador"]
    estimador = modelo.named_steps["modelo"]
    imputador_num = prep.named_transformers_["num"]["imputer"]
    imputador_cat = prep.named_transformers_["cat"]["imputer"]
    encoder = prep.named_transformers_["cat"]["encoder"]

    # Mismo orden que config.COLUMNAS_ENTRADA: primero categóricas, luego numéricas
    variables = [
        {"nombre": col, "tipo": "categorica", "dtype": "str",
         "imputacion": "moda", "valor_imputacion": str(moda),
         "categorias": [str(c) for c in categorias]}
        for col, moda, categorias in zip(config.COLUMNAS_CATEGORICAS, imputador_cat.statistics_, encoder.categories_)
    ]
    variables += [
        {"nombre": col, "tipo": "numerica", "dtype": "float",
         "imputacion": "mediana", "valor_imputacion": float(valor)}
        for col, valor in zip(config.COLUMNAS_NUMERICAS, imputador_num.statistics_)
    ]

    return {
        "version": version,
        "fecha_creacion": inicio.isoformat(timespec="seconds"),
        "tipo_problema": "clasificacion_binaria",
        "variable_objetivo": target,
        "etiquetas": {str(k): v for k, v in config.ETIQUETAS.items()},
        "algoritmo": type(estimador).__name__,
        "parametros_modelo": _serializable(estimador.get_params()),
        "semilla": semilla,
        "variables_entrada": variables,
        "variables_excluidas": {c: "posterior al evento (fuga de información)" for c in config.COLUMNAS_EXCLUIDAS},
        "columnas_ignoradas": ignoradas,
        "tratamiento_desconocidos": "categorías no vistas en el entrenamiento se codifican como ceros (OneHotEncoder handle_unknown='ignore')",
        "datos": {
            "fuente": config.FUENTE_DATOS,
            "archivo": ruta_datos.name,
            "sha256": _sha256(ruta_datos),
            "n_observaciones": int(len(df)),
            "n_columnas": int(df.shape[1]),
            "tasa_nulos_simulados": tasa_nulos,
        },
        "separacion": {
            "estrategia": "train_test_split aleatorio estratificado por la variable objetivo",
            "tamano_prueba": tamano_prueba,
            "n_entrenamiento": int(len(y_train)),
            "n_prueba": int(len(y_test)),
            "proporcion_clase1_entrenamiento": float(y_train.mean()),
            "proporcion_clase1_prueba": float(y_test.mean()),
            "validacion_cruzada": f"StratifiedKFold({folds_cv}) solo sobre entrenamiento" if folds_cv else None,
        },
        "metrica_evaluacion": config.METRICA_PRINCIPAL,
        "entorno": {
            "python": platform.python_version(),
            "librerias": {
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
            "archivo_dependencias": "requirements.txt",
        },
        "codigo": referencia_codigo(),
    }


def _serializable(params):
    return {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for k, v in params.items()}


def _sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()
