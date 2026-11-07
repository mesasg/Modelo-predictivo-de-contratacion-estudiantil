"""Esquema de los datos y valores por defecto heredados de la Fase 1."""

FUENTE_DATOS = "https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset"

TARGET = "placement_status"
ETIQUETAS = {0: "no contratado", 1: "contratado"}

# Variable posterior al evento: solo existe si el estudiante fue contratado (fuga de información)
COLUMNAS_EXCLUIDAS = ["salary_package_lpa"]

# Orden del esquema de entrada: el modelo siempre recibe las columnas en este orden
COLUMNAS_CATEGORICAS = ["branch", "college_tier"]
COLUMNAS_NUMERICAS = [
    "cgpa",
    "backlogs",
    "coding_skills",
    "dsa_score",
    "aptitude_score",
    "communication_skills",
    "ml_knowledge",
    "system_design",
    "internships",
    "projects_count",
    "certifications",
    "hackathons",
    "open_source_contributions",
    "extracurriculars",
]
COLUMNAS_ENTRADA = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS

SEMILLA = 42
TAMANO_PRUEBA = 0.20
FOLDS_CV = 5
# En la Fase 1 se simuló un 5 % de nulos en cada predictora para justificar la imputación
TASA_NULOS_SIMULADOS = 0.05

METRICA_PRINCIPAL = "roc_auc"

NOMBRE_MODELO = "modelo.joblib"
NOMBRE_METRICAS = "metrics.json"
NOMBRE_METADATOS = "metadata.json"
