# Modelo predictivo de contratación estudiantil

**Proyecto Integrador – Fase 1: Modelo predictivo**
Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-2 · Docente: Andrés Parra

## Integrantes
- Juan Manuel Tabares Uribe – [@juanmanuel-tabares](https://github.com/juanmanuel-tabares)
- Sara Mesa Gómez – [@mesasg](https://github.com/mesasg)

## Descripción del problema
Existe una alta incertidumbre en los estudiantes universitarios sobre su futuro laboral y sobre qué tan efectivos son sus esfuerzos académicos y extracurriculares para conseguir empleo. Este proyecto construye un modelo que predice si un estudiante será contratado y mide el peso real de factores como la institución, el promedio, las pasantías y las certificaciones.

## Fuente de los datos
[Student Placement Prediction Dataset](https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset) (Kaggle): 100.000 estudiantes y 16 variables predictoras. Datos sintéticos. El notebook lo descarga automáticamente con `kagglehub`.

## Objetivo del modelo
Predecir `placement_status` (1 = contratado, 0 = no contratado) a partir del desempeño académico, las habilidades técnicas y la experiencia extracurricular. Es un problema de **clasificación binaria supervisada**.

## Algoritmo utilizado
**Regresión Logística con pesos de clase** (`class_weight='balanced'`), dentro de un `Pipeline` con imputación (mediana y moda), One-Hot Encoding y `StandardScaler`. Se eligió con validación cruzada estratificada de 5 partes frente a Regresión Logística sin pesos, Random Forest y XGBoost: igualó o superó en ROC-AUC a los modelos de árboles y fue la que mejor detectó a los estudiantes no contratados.

## Métrica empleada
- **Principal:** ROC-AUC, porque no depende del umbral de decisión y no se ve afectada por el desbalance de clases (68 % contratados / 32 % no contratados).
- **Complementarias:** F1 y recall de la clase "no contratado", la clase minoritaria y la más relevante.
- La accuracy se reporta solo como referencia: predecir siempre "contratado" ya da 68 %.

## Principales resultados (conjunto de prueba)
| Modelo | ROC-AUC | F1 clase 0 | Recall clase 0 | Accuracy |
|---|---|---|---|---|
| Modelo base (Dummy) | 0,500 | 0,000 | 0,000 | 0,685 |
| Regresión Logística balanceada | **0,679** | **0,520** | **0,645** | 0,625 |

- El ROC-AUC en validación cruzada (0,680 ± 0,005) es prácticamente igual al de prueba: no hay sobreajuste.
- **Factores más influyentes:** el nivel de la institución (un estudiante de Tier-1 tiene ≈ 2,8 veces más probabilidad relativa de ser contratado que uno de Tier-3), el promedio (`cgpa`) y las pasantías. Las certificaciones pesan menos que la experiencia práctica, y `ml_knowledge`, `system_design` y `extracurriculars` no tienen efecto.
- Se excluyó `salary_package_lpa` porque solo existe si el estudiante fue contratado (fuga de información).

## Estructura
```
fase-1/
├── notebook.ipynb     # proceso completo: exploración, preparación, modelos, evaluación y almacenamiento
├── modelo.joblib      # pipeline entrenado (preprocesamiento + modelo)
├── requirements.txt   # dependencias con sus versiones
└── README.md
```

## Cómo ejecutar el notebook
**Google Colab:** abrir `notebook.ipynb` y usar *Entorno de ejecución → Reiniciar y ejecutar todo*. El dataset se descarga automáticamente y al final se genera `modelo.joblib`.

**Local:**
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Ejecutar todas las celdas en orden. La validación cruzada de la sección 8.1 tarda algunos minutos.

**Usar el modelo guardado:**
```python
import joblib
modelo = joblib.load("modelo.joblib")
# nuevos: DataFrame con las 16 columnas predictoras originales (datos crudos; admite valores faltantes)
prediccion = modelo.predict(nuevos)   # 1 = contratado, 0 = no contratado
```
