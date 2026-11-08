"""Almacenamiento versionado: models/<version>/{modelo.joblib, metrics.json, metadata.json}."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import joblib

from src.config import NOMBRE_METADATOS, NOMBRE_METRICAS, NOMBRE_MODELO

PATRON_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RAIZ_CODIGO = Path(__file__).resolve().parent.parent


class ErrorVersion(ValueError):
    """Identificador de versión inválido o ya existente."""


def version_automatica():
    """Identificador basado en la fecha y hora local, p. ej. v20261030-153012."""
    return datetime.now().strftime("v%Y%m%d-%H%M%S")


def validar_version(directorio_salida, version):
    """Rechaza identificadores con caracteres no permitidos o que ya existen (no se sobrescriben)."""
    if not PATRON_VERSION.match(version):
        raise ErrorVersion(
            f"Identificador de versión inválido: '{version}'. "
            "Use letras, números, '.', '_' o '-' (p. ej. v1, v2, v2026-10-30)."
        )
    destino = Path(directorio_salida) / version
    if destino.exists():
        raise ErrorVersion(
            f"La versión '{version}' ya existe en '{destino}'. "
            "Elija otro identificador; las versiones existentes no se sobrescriben."
        )
    return destino


def referencia_codigo():
    """Hash del commit de Git del código usado; si no hay Git (p. ej. en Docker), usa GIT_COMMIT."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=RAIZ_CODIGO,
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        estado = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=RAIZ_CODIGO,
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return {"git_commit": commit, "cambios_sin_commit": bool(estado), "origen": "git"}
    except (OSError, subprocess.SubprocessError):
        commit = os.environ.get("GIT_COMMIT")
        return {
            "git_commit": commit or "desconocido",
            "cambios_sin_commit": None,
            "origen": "variable GIT_COMMIT" if commit else "no disponible",
        }


def guardar_version(directorio_salida, version, modelo, metricas, metadatos):
    """Guarda los artefactos en una carpeta temporal y la renombra al final.

    Si algo falla a mitad del proceso no queda una versión incompleta que aparente ser válida.
    """
    directorio_salida = Path(directorio_salida)
    destino = validar_version(directorio_salida, version)
    directorio_salida.mkdir(parents=True, exist_ok=True)

    temporal = Path(tempfile.mkdtemp(prefix=f".{version}-tmp-", dir=directorio_salida))
    try:
        # mkdtemp crea la carpeta solo para su dueño (700); al renombrarla conservaría esos permisos
        # y una versión creada por root en Docker no se podría leer desde el equipo anfitrión
        temporal.chmod(0o755)
        joblib.dump(modelo, temporal / NOMBRE_MODELO)
        _escribir_json(temporal / NOMBRE_METRICAS, metricas)
        _escribir_json(temporal / NOMBRE_METADATOS, metadatos)
        # Se vuelve a comprobar justo antes de renombrar por si otra ejecución creó la versión
        validar_version(directorio_salida, version)
        temporal.rename(destino)
    except BaseException:
        shutil.rmtree(temporal, ignore_errors=True)
        raise

    return {
        "directorio": destino,
        "modelo": destino / NOMBRE_MODELO,
        "metricas": destino / NOMBRE_METRICAS,
        "metadatos": destino / NOMBRE_METADATOS,
    }


def _escribir_json(ruta, contenido):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)
