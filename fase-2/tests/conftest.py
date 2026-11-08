"""Datos sintéticos pequeños y una versión entrenada una sola vez para todas las pruebas.

No se descarga nada ni se usa la carpeta models/ de la entrega: todo se escribe en carpetas temporales.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.training import entrenar

N_FILAS = 300
RAMAS = ["CE", "CSE", "Chemical", "ECE", "EE", "IT", "ME"]
NIVELES = ["Tier-1", "Tier-2", "Tier-3"]


def generar_datos(n, semilla):
    """Estudiantes con el esquema real del proyecto; la contratación depende de cgpa, pasantías y nivel."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({
        "branch": rng.choice(RAMAS, n),
        "college_tier": rng.choice(NIVELES, n),
        "cgpa": rng.uniform(4, 10, n).round(2),
        "backlogs": rng.integers(0, 5, n),
        "coding_skills": rng.uniform(0, 10, n).round(1),
        "dsa_score": rng.uniform(0, 10, n).round(1),
        "aptitude_score": rng.uniform(20, 100, n).round(1),
        "communication_skills": rng.uniform(0, 10, n).round(1),
        "ml_knowledge": rng.uniform(0, 10, n).round(1),
        "system_design": rng.uniform(0, 10, n).round(1),
        "internships": rng.integers(0, 4, n),
        "projects_count": rng.integers(0, 6, n),
        "certifications": rng.integers(0, 6, n),
        "hackathons": rng.integers(0, 4, n),
        "open_source_contributions": rng.integers(0, 4, n),
        "extracurriculars": rng.integers(0, 4, n),
    })
    logit = (1.5 * (df["cgpa"] - 7) + 0.8 * (df["internships"] - 1.5)
             + df["college_tier"].map({"Tier-1": 1.0, "Tier-2": 0.0, "Tier-3": -1.0}))
    df[config.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # Variable posterior al evento, como en el dataset real: solo existe si hubo contratación
    df["salary_package_lpa"] = np.where(df[config.TARGET] == 1, rng.uniform(3, 20, n).round(2), np.nan)
    return df


@pytest.fixture(scope="session")
def ruta_datos(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("data") / "datos.csv"
    generar_datos(N_FILAS, semilla=0).to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def version_entrenada(tmp_path_factory, ruta_datos):
    """Se entrena una sola vez; las pruebas de predicción reutilizan esta versión."""
    return entrenar(ruta_datos, tmp_path_factory.mktemp("models"), "v1", semilla=42, folds_cv=3)


@pytest.fixture
def entrada():
    """Estudiantes nuevos sin la variable objetivo ni el salario, como en data/ejemplo_prediccion.csv."""
    return generar_datos(20, semilla=1).drop(columns=[config.TARGET, "salary_package_lpa"])
