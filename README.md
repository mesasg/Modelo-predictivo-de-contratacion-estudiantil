# Modelo predictivo de contratación estudiantil

Proyecto Integrador – Fase 1 · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-2

## Integrantes
- Juan Manuel Tabares Uribe
- Sara Mesa Gómez

## Descripción del problema
Muchos estudiantes universitarios no saben qué tan efectivos son sus esfuerzos académicos y extracurriculares para conseguir empleo al graduarse. Este proyecto busca medir el peso real de factores como la institución, el promedio, las pasantías y las certificaciones en la contratación.

## Fuente del conjunto de datos
[Student Placement Prediction Dataset](https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset) (Kaggle): 100.000 estudiantes y 16 variables predictoras (datos sintéticos).

## Objetivo del modelo
Predecir si un estudiante será contratado (`placement_status`: 1 = sí, 0 = no) e identificar qué factores pesan más. Es un problema de clasificación binaria supervisada.

## Algoritmo utilizado
**Regresión Logística con pesos de clase** (`class_weight='balanced'`), dentro de un `Pipeline` con imputación, One-Hot Encoding y estandarización. Se eligió con validación cruzada de 5 partes frente a Regresión Logística sin pesos, Random Forest y XGBoost.

## Métrica empleada
- **ROC-AUC** (principal): no depende del umbral y no se ve afectada por el desbalance de clases (68 % contratados).
- **F1 y recall de la clase "no contratado"** (complementarias): miden la detección de los estudiantes en riesgo.

## Principales resultados
| Modelo (conjunto de prueba) | ROC-AUC | F1 clase 0 | Recall clase 0 |
|---|---|---|---|
| Modelo base (Dummy) | 0,500 | 0,000 | 0,000 |
| Regresión Logística balanceada | **0,679** | **0,520** | **0,645** |

- El modelo supera claramente al modelo base y detecta el 64,5 % de los no contratados. El resultado es estable y sin sobreajuste (ROC-AUC en validación cruzada: 0,680 ± 0,005).
- Los factores más influyentes son el nivel de la institución, el promedio y las pasantías. Las certificaciones pesan menos que la experiencia práctica.

## Instrucciones para ejecutar el notebook
1. Abrir `notebook.ipynb` en Google Colab y usar *Entorno de ejecución → Reiniciar y ejecutar todo*. El dataset se descarga automáticamente con `kagglehub`.
2. Para ejecutarlo localmente: `pip install -r requirements.txt` y luego `jupyter notebook notebook.ipynb`.
3. Al terminar se genera `modelo.joblib`. Para usarlo:
   ```python
   import joblib
   modelo = joblib.load("modelo.joblib")
   modelo.predict(nuevos)  # nuevos: DataFrame con las 16 columnas predictoras originales
   ```