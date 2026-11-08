# Modelo predictivo de contratación estudiantil

**Proyecto Integrador – Fase 1: Modelo predictivo**
Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-2 · Docente: Andrés Parra

## Integrantes
- Juan Manuel Tabares Uribe – [@juanmanuel-tabares](https://github.com/juanmanuel-tabares)
- Sara Mesa Gómez – [@mesasg](https://github.com/mesasg)

## Descripción del problema
Existe una alta incertidumbre en los estudiantes universitarios sobre su futuro laboral y sobre qué tan efectivos son sus esfuerzos académicos y extracurriculares para conseguir empleo. Este proyecto construye un modelo que predice si un estudiante será contratado y estima qué tan asociados están con la contratación, dentro del modelo, factores como la institución, el promedio, las pasantías y las certificaciones.

## Fuente de los datos
[Student Placement Prediction Dataset](https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset) (Kaggle): 100.000 estudiantes y 16 variables predictoras. Datos sintéticos. El notebook lo descarga automáticamente con `kagglehub`.

Las predictoras no traen faltantes, así que, según lo acordado con el profesor, se simuló un 5 % de nulos en cada una. Esto deja cerca del 56 % de las filas con al menos un faltante (1 − 0,95¹⁶), más de lo habitual en datos reales, así que funciona como una prueba de robustez de la imputación.

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

- **Matriz de confusión:** de los 6.305 no contratados de prueba, el modelo identifica 4.067 y deja 2.238 sin detectar. Además clasifica como no contratados a 5.264 estudiantes que sí lo fueron, así que sus alertas aciertan el 43,6 % de las veces (precisión de la clase 0).
- El ROC-AUC en validación cruzada (0,680 ± 0,005) es prácticamente igual al de prueba. Esto indica estabilidad en esta distribución sintética, no generalización demostrada al mercado laboral real.
- **Variables con mayor asociación en el modelo:** el nivel de la institución, el promedio (`cgpa`) y las pasantías. Las *odds* de contratación de un estudiante de Tier-1 son ≈ 2,84 veces las de uno de Tier-3 con las demás variables iguales (exp(0,581 − (−0,462))). Esto compara *odds*, no probabilidades: no significa que su probabilidad sea 2,84 veces mayor. Las certificaciones tienen una asociación menor que la experiencia práctica, y `ml_knowledge`, `system_design` y `extracurriculars` tienen coeficientes cercanos a cero, es decir, poca asociación condicional en este ajuste.
- Se excluyó `salary_package_lpa` porque solo existe si el estudiante fue contratado (fuga de información).

## Alcance y limitaciones
- Los datos son sintéticos: los coeficientes describen asociaciones del modelo con este dataset, no efectos causales. Antes de presentar los resultados como evidencia del mercado laboral habría que validar el enfoque con datos reales de egresados, describiendo su población y periodo.
- Por usar `class_weight='balanced'`, `predict_proba` entrega un puntaje para ordenar estudiantes, no una probabilidad calibrada de contratación.
- El umbral de decisión es 0,5. Para usarlo como herramienta de orientación habría que definir el costo de cada tipo de error y ajustar el umbral únicamente con validación, nunca con el conjunto de prueba.

## Estructura
```
fase-1/
├── Fase_I_Modelos.ipynb   # proceso completo: exploración, preparación, modelos, evaluación y almacenamiento
├── modelo.joblib          # pipeline entrenado (preprocesamiento + modelo)
├── requirements.txt       # dependencias con sus versiones
└── README.md
```

## Cómo ejecutar el notebook
**Google Colab:** subir `fase-1/Fase_I_Modelos.ipynb` y usar *Entorno de ejecución → Reiniciar y ejecutar todo*. El dataset se descarga automáticamente y al final se genera `modelo.joblib`.

**Local** (probado con Python 3.13; comandos para Bash desde la raíz del repositorio):
```bash
cd fase-1
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install notebook             # solo para abrir el notebook en Jupyter
jupyter notebook Fase_I_Modelos.ipynb
```
Ejecutar todas las celdas en orden. La validación cruzada de la sección 8.1 tarda algunos minutos. El modelo se guarda en `fase-1/modelo.joblib`.

**Usar el modelo guardado** (desde `fase-1`, con la misma versión de scikit-learn de `requirements.txt`):
```python
import joblib
modelo = joblib.load("modelo.joblib")
# nuevos: DataFrame con las 16 columnas predictoras originales (datos crudos; admite valores faltantes)
prediccion = modelo.predict(nuevos)   # 1 = contratado, 0 = no contratado
```

## Correcciones posteriores a la calificación
- `requirements.txt`: versiones explícitas de `joblib` y `kagglehub` y `scipy` fijado en 1.15.3, compatible con scikit-learn 1.6.1.
- README: cambio del nombre del notebook (`Fase_I_Modelos.ipynb`) y ejecución desde `fase-1`.
- Interpretación (notebook, secciones 1, 3 y 8 a 10, y README): los odds ratios se explican como cocientes de *odds* y no de probabilidades, y las conclusiones se limitan a asociaciones del modelo con un dataset sintético. Se precisaron las cifras de la matriz de confusión y el alcance de los nulos simulados (≈ 56 % de las filas).
- Los cambios solo fueron de explicaciones/analisis, no cambió el código, el modelo ni las métricas.
