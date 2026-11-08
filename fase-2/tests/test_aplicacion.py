"""Pruebas de la aplicación: entrenamiento, predicción, valores faltantes, entradas inválidas y versionamiento."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import train
from src import config
from src.data import ErrorDatos
from src.prediction import cargar_version, guardar_predicciones, predecir
from src.storage import ErrorVersion
from src.training import entrenar
from tests.conftest import N_FILAS

RAIZ = Path(__file__).resolve().parent.parent


def _leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _huella(carpeta):
    """Contenido de cada archivo de una carpeta, para comprobar que no se modificó."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(carpeta).iterdir())}


# 1. Entrenamiento y almacenamiento

def test_entrenamiento_genera_artefacto_metricas_y_metadatos(version_entrenada):
    rutas = version_entrenada["rutas"]
    for clave in ("modelo", "metricas", "metadatos"):
        assert rutas[clave].is_file()

    # El artefacto es el pipeline completo y recibe las 16 variables crudas (sin objetivo ni salario)
    modelo = joblib.load(rutas["modelo"])
    assert list(modelo.named_steps) == ["preprocesador", "modelo"]
    assert list(modelo.feature_names_in_) == config.COLUMNAS_ENTRADA

    metricas = _leer_json(rutas["metricas"])
    assert metricas["modelo_base"]["roc_auc"] == 0.5
    assert metricas["modelo"]["roc_auc"] > metricas["modelo_base"]["roc_auc"]
    assert metricas["validacion_cruzada"]["folds"] == 3

    metadatos = _leer_json(rutas["metadatos"])
    assert metadatos["version"] == "v1"
    assert metadatos["variable_objetivo"] == config.TARGET
    assert metadatos["algoritmo"] == "LogisticRegression"
    assert metadatos["semilla"] == 42
    assert [v["nombre"] for v in metadatos["variables_entrada"]] == config.COLUMNAS_ENTRADA
    assert "salary_package_lpa" in metadatos["variables_excluidas"]
    assert metadatos["datos"]["n_observaciones"] == N_FILAS
    sep = metadatos["separacion"]
    assert sep["n_entrenamiento"] + sep["n_prueba"] == N_FILAS
    assert metadatos["codigo"]["git_commit"]


# 2. Carga y predicción

def test_prediccion_conserva_los_resultados_del_modelo_entrenado(version_entrenada, entrada):
    carpeta = version_entrenada["rutas"]["directorio"]
    huella_antes = _huella(carpeta)

    modelo, metadatos, _ = cargar_version(carpeta)
    resultados, _ = predecir(modelo, entrada, metadatos)

    # Una predicción por fila, en el orden de la entrada
    assert len(resultados) == len(entrada)
    assert resultados["fila"].tolist() == list(range(len(entrada)))
    assert set(resultados["etiqueta"]) <= {"contratado", "no contratado"}

    # El modelo cargado reproduce al modelo en memoria antes de guardarlo
    original = version_entrenada["modelo"]
    np.testing.assert_allclose(
        resultados["puntaje_contratado"], original.predict_proba(entrada)[:, 1], atol=1e-6
    )
    assert resultados["prediccion"].tolist() == original.predict(entrada).tolist()

    # Predecir no reentrena ni reescribe la versión
    assert _huella(carpeta) == huella_antes


def test_orden_de_columnas_y_columnas_adicionales_no_cambian_la_prediccion(version_entrenada, entrada):
    modelo, metadatos, _ = cargar_version(version_entrenada["rutas"]["modelo"])
    esperado, _ = predecir(modelo, entrada, metadatos)

    variante = entrada[entrada.columns[::-1]].assign(extra=1, placement_status=0)
    resultados, avisos = predecir(modelo, variante, metadatos)

    pd.testing.assert_frame_equal(resultados, esperado)
    assert any("extra" in a and "placement_status" in a for a in avisos)


# 3. Valores faltantes

def test_valores_faltantes_se_imputan_con_lo_aprendido_en_entrenamiento(version_entrenada, entrada):
    modelo, metadatos, _ = cargar_version(version_entrenada["rutas"]["directorio"])
    imputacion = {v["nombre"]: v["valor_imputacion"] for v in metadatos["variables_entrada"]}

    con_nulos = entrada.copy()
    con_nulos.loc[0, "cgpa"] = np.nan
    con_nulos.loc[1, "college_tier"] = np.nan
    resultados, avisos = predecir(modelo, con_nulos, metadatos)

    assert len(resultados) == len(entrada)
    assert resultados["puntaje_contratado"].notna().all()
    assert any("2 celdas vacías" in a for a in avisos)

    # Una celda vacía equivale a la mediana o la moda guardadas en metadata.json
    rellenado = entrada.copy()
    rellenado.loc[0, "cgpa"] = imputacion["cgpa"]
    rellenado.loc[1, "college_tier"] = imputacion["college_tier"]
    esperado, _ = predecir(modelo, rellenado, metadatos)
    pd.testing.assert_frame_equal(resultados, esperado)


def test_celdas_vacias_en_el_csv_se_procesan_desde_la_linea_de_comandos(version_entrenada, entrada, tmp_path):
    ruta_entrada = tmp_path / "con_vacios.csv"
    entrada.assign(cgpa=entrada["cgpa"].where(entrada.index != 0)).to_csv(ruta_entrada, index=False)
    salida = tmp_path / "predicciones.csv"

    proceso = _ejecutar_predict(version_entrenada["rutas"]["directorio"], ruta_entrada, salida)

    assert proceso.returncode == 0, proceso.stderr
    assert len(pd.read_csv(salida)) == len(entrada)


# 4. Entradas inválidas

def _ejecutar_predict(modelo, entrada, salida):
    return subprocess.run(
        [sys.executable, "predict.py", "--model", str(modelo), "--input", str(entrada), "--output", str(salida)],
        cwd=RAIZ, capture_output=True, text=True,
    )


def test_columna_ausente_termina_con_error_y_sin_archivo_de_salida(version_entrenada, entrada, tmp_path):
    ruta_entrada = tmp_path / "sin_cgpa.csv"
    entrada.drop(columns=["cgpa"]).to_csv(ruta_entrada, index=False)
    salida = tmp_path / "predicciones.csv"

    proceso = _ejecutar_predict(version_entrenada["rutas"]["directorio"], ruta_entrada, salida)

    assert proceso.returncode != 0
    assert "Faltan columnas obligatorias: cgpa" in proceso.stderr
    assert not salida.exists()


@pytest.mark.parametrize("columna, valor, mensaje", [
    ("cgpa", "ocho", "debe ser numérica"),
    ("cgpa", 15, "fuera del rango"),
    ("aptitude_score", -1, "fuera del rango"),
    ("internships", 1.5, "debe ser entera"),
])
def test_tipos_y_rangos_invalidos_se_rechazan(version_entrenada, entrada, columna, valor, mensaje):
    modelo, metadatos, _ = cargar_version(version_entrenada["rutas"]["directorio"])
    invalida = entrada.astype({columna: object})
    invalida.loc[3, columna] = valor

    with pytest.raises(ErrorDatos, match=mensaje):
        predecir(modelo, invalida, metadatos)


def test_categoria_no_vista_se_predice_y_se_informa(version_entrenada, entrada):
    modelo, metadatos, _ = cargar_version(version_entrenada["rutas"]["directorio"])
    nueva = entrada.copy()
    nueva.loc[0, "branch"] = "Medicina"

    resultados, avisos = predecir(modelo, nueva, metadatos)

    assert len(resultados) == len(entrada)
    assert any("Medicina" in a for a in avisos)


def test_modelo_inexistente_termina_con_error(entrada, tmp_path):
    ruta_entrada = tmp_path / "entrada.csv"
    entrada.to_csv(ruta_entrada, index=False)
    salida = tmp_path / "predicciones.csv"

    proceso = _ejecutar_predict(tmp_path / "models" / "v9", ruta_entrada, salida)

    assert proceso.returncode != 0
    assert "No se encontró el modelo" in proceso.stderr
    assert not salida.exists()


# 5. Versionamiento

def test_nueva_version_conserva_la_anterior_y_no_se_sobrescribe(ruta_datos, tmp_path):
    modelos = tmp_path / "models"
    entrenar(ruta_datos, modelos, "v1", folds_cv=0)
    huella_v1 = _huella(modelos / "v1")

    entrenar(ruta_datos, modelos, "v2", folds_cv=0, C=0.1)

    # Un identificador repetido se rechaza, tanto desde la función como desde la línea de comandos
    with pytest.raises(ErrorVersion, match="ya existe"):
        entrenar(ruta_datos, modelos, "v1", folds_cv=0, C=0.5)
    codigo = train.main(["--data", str(ruta_datos), "--output-dir", str(modelos), "--version", "v1"])
    assert codigo != 0

    assert _huella(modelos / "v1") == huella_v1
    # Sin carpetas temporales ni versiones incompletas
    assert sorted(p.name for p in modelos.iterdir()) == ["v1", "v2"]

    for version, C in (("v1", 1.0), ("v2", 0.1)):
        modelo, metadatos, _ = cargar_version(modelos / version)
        assert metadatos["version"] == version
        assert metadatos["parametros_modelo"]["C"] == C
        assert hasattr(modelo, "predict_proba")


@pytest.mark.skipif(os.name == "nt", reason="los permisos POSIX no aplican en Windows")
def test_version_y_predicciones_son_legibles_fuera_del_contenedor(version_entrenada, entrada, tmp_path):
    assert version_entrenada["rutas"]["directorio"].stat().st_mode & 0o777 == 0o755

    modelo, metadatos, _ = cargar_version(version_entrenada["rutas"]["directorio"])
    salida = guardar_predicciones(predecir(modelo, entrada, metadatos)[0], tmp_path / "predicciones.csv")
    assert salida.stat().st_mode & 0o777 == 0o644


# Línea de comandos

@pytest.mark.parametrize("script", ["train.py", "predict.py"])
def test_ayuda_disponible(script):
    proceso = subprocess.run([sys.executable, script, "--help"], cwd=RAIZ, capture_output=True, text=True)
    assert proceso.returncode == 0
    assert "usage" in proceso.stdout
