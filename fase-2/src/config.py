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

# Rangos válidos de cada escala (mínimo, máximo); None = sin límite superior.
# Son los límites del dominio, no los valores observados, para no rechazar perfiles plausibles.
RANGOS = {
    "cgpa": (0, 10),
    "coding_skills": (0, 10),
    "dsa_score": (0, 10),
    "aptitude_score": (0, 100),
    "communication_skills": (0, 10),
    "ml_knowledge": (0, 10),
    "system_design": (0, 10),
    "backlogs": (0, None),
    "internships": (0, None),
    "projects_count": (0, None),
    "certifications": (0, None),
    "hackathons": (0, None),
    "open_source_contributions": (0, None),
    "extracurriculars": (0, None),
}
# Variables de conteo: deben ser números enteros
COLUMNAS_CONTEO = [
    "backlogs",
    "internships",
    "projects_count",
    "certifications",
    "hackathons",
    "open_source_contributions",
    "extracurriculars",
]

SEMILLA = 42
TAMANO_PRUEBA = 0.20
FOLDS_CV = 5
# En la Fase 1 se simuló un 5 % de nulos en cada predictora para justificar la imputación
TASA_NULOS_SIMULADOS = 0.05

METRICA_PRINCIPAL = "roc_auc"
# predict() usa el umbral 0,5 sobre el puntaje de la clase 1
UMBRAL_DECISION = 0.5

# Alcance de los resultados (retroalimentación de la Fase 1)
ADVERTENCIAS = [
    "Datos sintéticos: los coeficientes describen asociaciones del modelo con este dataset, no efectos causales "
    "del mercado laboral; validar con datos reales de egresados antes de usarlos como evidencia.",
    "Con class_weight='balanced', predict_proba entrega un puntaje para ordenar estudiantes, no una probabilidad calibrada.",
    "El umbral 0,5 no se ajustó a un costo de errores; si se cambia, debe elegirse solo con validación, nunca con prueba.",
]

NOMBRE_MODELO = "modelo.joblib"
NOMBRE_METRICAS = "metrics.json"
NOMBRE_METADATOS = "metadata.json"
