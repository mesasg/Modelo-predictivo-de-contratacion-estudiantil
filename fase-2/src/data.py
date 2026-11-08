"""Lectura y validación de los datos."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COLUMNAS_CATEGORICAS, COLUMNAS_CONTEO, COLUMNAS_ENTRADA, COLUMNAS_NUMERICAS, RANGOS


class ErrorDatos(ValueError):
    """Error previsible en el archivo de datos (columnas ausentes, tipos incompatibles, etc.)."""


def leer_csv(ruta):
    """Lee un CSV y falla con un mensaje claro si no existe, está vacío o no se puede leer."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")
    try:
        df = pd.read_csv(ruta)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise ErrorDatos(f"No se pudo leer '{ruta}' como CSV: {e}") from e
    if df.empty:
        raise ErrorDatos(f"El archivo '{ruta}' no contiene filas.")
    return df


def preparar_variables(df):
    """Valida y ordena las variables de entrada según el esquema del modelo.

    - Falla si falta alguna columna obligatoria (una celda nula sí se admite).
    - Ignora columnas adicionales y las devuelve para informarlas.
    - Convierte las numéricas a número y las categóricas a texto, conservando los nulos.
    - Falla si una numérica está fuera de su rango o si un conteo no es entero.
    """
    faltantes = [c for c in COLUMNAS_ENTRADA if c not in df.columns]
    if faltantes:
        raise ErrorDatos(f"Faltan columnas obligatorias: {', '.join(faltantes)}")

    X = df[COLUMNAS_ENTRADA].copy()

    for col in COLUMNAS_NUMERICAS:
        convertida = pd.to_numeric(X[col], errors="coerce")
        invalidos = convertida.isna() & X[col].notna()
        if invalidos.any():
            ejemplos = X.loc[invalidos, col].astype(str).unique()[:3].tolist()
            raise ErrorDatos(
                f"La columna '{col}' debe ser numérica; valores no válidos: {ejemplos}"
            )
        X[col] = convertida.astype(float)
        _validar_rango(X[col], col)

    for col in COLUMNAS_CATEGORICAS:
        # Texto en las celdas con valor y np.nan en las vacías (el imputador no reconoce None)
        valores = X[col].astype(object)
        X[col] = valores.where(valores.isna(), valores.astype(str)).where(valores.notna(), np.nan)

    adicionales = [c for c in df.columns if c not in COLUMNAS_ENTRADA]
    return X, adicionales


def _validar_rango(valores, col):
    """Los nulos se admiten (se imputan); los valores presentes deben estar dentro del dominio."""
    minimo, maximo = RANGOS[col]
    fuera = valores < minimo
    if maximo is not None:
        fuera |= valores > maximo
    if fuera.any():
        limite = f"[{minimo}, {maximo}]" if maximo is not None else f">= {minimo}"
        ejemplos = valores[fuera].unique()[:3].tolist()
        raise ErrorDatos(
            f"La columna '{col}' tiene {int(fuera.sum())} valores fuera del rango {limite}: {ejemplos}"
        )
    if col in COLUMNAS_CONTEO:
        no_enteros = valores.notna() & (valores % 1 != 0)
        if no_enteros.any():
            ejemplos = valores[no_enteros].unique()[:3].tolist()
            raise ErrorDatos(f"La columna '{col}' es un conteo y debe ser entera; valores no válidos: {ejemplos}")


def preparar_objetivo(df, target):
    """Valida la variable objetivo: debe existir, no tener nulos y ser binaria (0/1)."""
    if target not in df.columns:
        raise ErrorDatos(f"No se encontró la variable objetivo '{target}' en los datos.")
    y = df[target]
    if y.isna().any():
        raise ErrorDatos(f"La variable objetivo '{target}' tiene {int(y.isna().sum())} valores nulos.")
    y = pd.to_numeric(y, errors="coerce")
    valores = set(y.dropna().unique())
    if y.isna().any() or not valores <= {0, 1}:
        raise ErrorDatos(f"La variable objetivo '{target}' debe contener solo 0 y 1.")
    if len(valores) < 2:
        raise ErrorDatos(f"La variable objetivo '{target}' solo tiene una clase; no se puede entrenar.")
    return y.astype(int)


def simular_nulos(X, tasa, semilla):
    """Reemplaza al azar una proporción de celdas por nulos, como en la Fase 1.

    La máscara no depende de la variable objetivo, así que no introduce fuga de información.
    """
    if tasa <= 0:
        return X
    mascara = np.random.RandomState(semilla).rand(*X.shape) < tasa
    return X.mask(mascara)
