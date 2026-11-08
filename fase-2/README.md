# Modelo predictivo de contratación estudiantil

**Proyecto Integrador – Fase 2: Scripts y Docker**
Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-2 · Docente: Andrés Parra

## Integrantes
- Juan Manuel Tabares Uribe – [@juanmanuel-tabares](https://github.com/juanmanuel-tabares)
- Sara Mesa Gómez – [@mesasg](https://github.com/mesasg)

## Problema y relación con la Fase 1
El modelo predice si un estudiante universitario será contratado (`placement_status`: 1 = contratado, 0 = no contratado) a partir de su desempeño académico, habilidades técnicas y experiencia extracurricular. Es un problema de **clasificación binaria supervisada**.

En la [Fase 1](../fase-1/README.md) el proceso completo estaba en un notebook. En esta fase se convierte en una aplicación reutilizable: `train.py` entrena y guarda versiones del modelo y `predict.py` carga una versión y predice sobre datos nuevos, ambos desde la línea de comandos y dentro de Docker. Se conservan el problema, los datos y el modelo de la Fase 1: Regresión Logística con `class_weight='balanced'` dentro de un `Pipeline` con imputación (mediana y moda), `StandardScaler` y One-Hot Encoding. `salary_package_lpa` sigue excluida porque solo existe si el estudiante fue contratado (fuga de información).

## Correcciones respecto a la Fase 1
**En `fase-1/`** (no cambió el código, el modelo ni las métricas):
- `requirements.txt`: versiones explícitas de `joblib` (1.6.0) y `kagglehub` (1.0.3), antes `X.X.X`, y `scipy==1.15.3`, compatible con scikit-learn 1.6.1. Se comprobó `pip install -r requirements.txt` en un entorno limpio.
- README: nombre real del notebook (`Fase_I_Modelos.ipynb`) y comandos de ejecución desde `fase-1`.
- Interpretación (notebook y README): los odds ratios comparan *odds*, no probabilidades. Las *odds* de un estudiante de Tier-1 son ≈ 2,84 veces las de uno de Tier-3 con las demás variables iguales, lo que no significa 2,84 veces más probabilidad. Los coeficientes se presentan como asociaciones condicionales en un dataset sintético, no como efectos causales del mercado laboral.
- Se precisaron las cifras de la matriz de confusión (4.067 no contratados detectados, 2.238 sin detectar, 5.264 falsas alertas) y el alcance de los nulos simulados (≈ 56 % de las filas, una prueba de robustez).

**En la Fase 2:**
- Se agregó la precisión de la clase 0 a las métricas.
- Pensando en la API, se validan tipos, rangos y conteos enteros, y se informan las categorías no vistas en el entrenamiento.
- `metadata.json` registra las advertencias de alcance: datos sintéticos, puntajes no calibrados y umbral 0,5 sin ajustar a costos.
- Cada Pull Request hacia `develop` lo revisa y aprueba el otro integrante antes del merge.

## Estructura
```
fase-2/
├── train.py                  # punto de entrada: entrenamiento
├── predict.py                # punto de entrada: predicción
├── src/
│   ├── config.py             # esquema de los datos, rangos válidos y valores por defecto
│   ├── data.py               # lectura y validación de datos
│   ├── pipeline.py           # preprocesamiento, modelo y modelo base
│   ├── evaluation.py         # métricas y validación cruzada
│   ├── storage.py            # versiones, guardado atómico y referencia al commit
│   ├── training.py           # proceso completo de entrenamiento
│   └── prediction.py         # carga de versiones y predicción
├── data/
│   └── ejemplo_prediccion.csv  # 10 estudiantes sin la variable objetivo
├── models/
│   ├── v1/                   # modelo.joblib, metrics.json, metadata.json
│   └── v2/
├── outputs/                  # predicciones generadas (no se suben a Git)
├── tests/
│   ├── conftest.py           # datos sintéticos pequeños y una versión entrenada una sola vez
│   └── test_aplicacion.py    # pruebas con pytest
├── requirements.txt
├── pytest.ini                # configuración de pytest (carpeta tests/ y rutas de importación)
├── Dockerfile                # imagen python:3.13-slim; no entrena durante la construcción
├── .dockerignore
└── README.md
```

## Instalación
Versión de Python: **3.13** (probado con 3.13.14). La imagen de Docker usa la misma versión.

Terminal usada: **PowerShell** en Windows. Todos los comandos se ejecutan desde la carpeta `fase-2`.
```powershell
cd fase-2
python -m venv .venv
.venv\Scripts\Activate.ps1          # si se bloquea: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```
En Bash (Linux, macOS o Git Bash), la activación es `source .venv/bin/activate` (en Git Bash: `source .venv/Scripts/activate`).

`requirements.txt` fija las versiones exactas de numpy, pandas, scikit-learn, scipy, joblib y pytest. Los modelos guardados deben cargarse con la misma versión de scikit-learn (1.6.1).

## Datos
- **Fuente:** [Student Placement Prediction Dataset](https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset) (Kaggle). 100.000 estudiantes, 16 predictoras, datos sintéticos.
- **Ubicación esperada:** `fase-2/data/student_placement_synthetic.csv`. El archivo completo no se sube a Git (lo excluye el `.gitignore`); solo se versiona `data/ejemplo_prediccion.csv`.
- **Opción A, descarga manual:** descargar el zip desde Kaggle, descomprimirlo y copiar `student_placement_synthetic.csv` a `fase-2/data/`.
- **Opción B, kagglehub** (no requiere credenciales):
  ```powershell
  pip install kagglehub==1.0.3
  python -c "import kagglehub, shutil, pathlib; p = pathlib.Path(kagglehub.dataset_download('suhanigupta04/student-placement-prediction-dataset')); shutil.copy(p / 'student_placement_synthetic.csv', 'data/')"
  ```
  `kagglehub` solo se usa para descargar; no está en `requirements.txt` porque la aplicación no lo necesita.

## Entrenamiento: `train.py`
```powershell
python train.py --help
python train.py --data data/student_placement_synthetic.csv --target placement_status --output-dir models --version v1 --seed 42
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--data` | (obligatorio) | CSV con datos etiquetados |
| `--target` | `placement_status` | variable objetivo (binaria 0/1) |
| `--output-dir` | `models` | carpeta donde se crean las versiones |
| `--version` | fecha y hora | identificador de la versión (p. ej. `v1`) |
| `--seed` | `42` | semilla de la separación, los nulos simulados, la validación cruzada y el modelo |
| `--test-size` | `0.2` | proporción reservada para prueba |
| `--cv-folds` | `5` | partes de la validación cruzada sobre el entrenamiento (0 = omitir) |
| `--missing-rate` | `0.05` | proporción de celdas que se vuelven nulas, como en la Fase 1 (0 = no simular) |
| `--C` | `1.0` | inverso de la regularización de la Regresión Logística |
| `--max-iter` | `1000` | iteraciones máximas del optimizador |

**Proceso:**
1. Lee el CSV y valida la variable objetivo y las 16 columnas (tipos, rangos y conteos enteros).
2. Simula el 5 % de nulos por predictora, como se acordó en la Fase 1.
3. Separa 80/20 de forma estratificada **antes** de ajustar cualquier transformación. Cada fila es un estudiante distinto, así que no hay grupos que mantener juntos.
4. Hace la validación cruzada estratificada solo con el entrenamiento; el pipeline se reajusta en cada partición.
5. Entrena el modelo final y el modelo base (`DummyClassifier`, clase más frecuente) con el mismo entrenamiento y evalúa ambos con el mismo conjunto de prueba.
6. Guarda la versión.

**Versionamiento:** cada versión queda en `models/<version>/`:
- `modelo.joblib`: pipeline completo (preprocesamiento ajustado + modelo). Recibe los datos crudos.
- `metrics.json`: métricas en prueba del modelo y del modelo base (ROC-AUC, F1, recall y precisión de la clase 0, accuracy, matriz de confusión) y validación cruzada.
- `metadata.json`: versión, fecha, tipo de problema, objetivo, etiquetas, algoritmo y parámetros, semilla, variables de entrada (tipo, rango, categorías y valor de imputación), datos usados (archivo, sha256, observaciones), separación, métrica, entorno (Python y librerías), `requirements.txt` y hash del commit.

Una versión existente **nunca se sobrescribe**: repetir un identificador termina con error y código 1 sin modificar nada. Los archivos se escriben primero en una carpeta temporal que se renombra al final, así que una ejecución fallida no deja versiones incompletas.

## Predicción: `predict.py`
```powershell
python predict.py --help
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
python predict.py --model models/v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--model` | (obligatorio) | `modelo.joblib` o carpeta de la versión (`models/v1`) |
| `--input` | (obligatorio) | CSV con las 16 variables de entrada |
| `--output` | `outputs/predicciones.csv` | CSV de salida (se crea la carpeta si no existe) |

`predict.py` solo carga el pipeline guardado: no reentrena ni reajusta transformaciones.

### Esquema del CSV de entrada
Se requieren las 16 columnas, en cualquier orden. La variable objetivo no es necesaria. Todas admiten celdas vacías.

| Columna | Tipo | Valores válidos |
|---|---|---|
| `branch` | categórica | CE, CSE, Chemical, ECE, EE, IT, ME |
| `college_tier` | categórica | Tier-1, Tier-2, Tier-3 |
| `cgpa` | numérica | 0 a 10 |
| `backlogs` | entera | ≥ 0 |
| `coding_skills` | numérica | 0 a 10 |
| `dsa_score` | numérica | 0 a 10 |
| `aptitude_score` | numérica | 0 a 100 |
| `communication_skills` | numérica | 0 a 10 |
| `ml_knowledge` | numérica | 0 a 10 |
| `system_design` | numérica | 0 a 10 |
| `internships` | entera | ≥ 0 |
| `projects_count` | entera | ≥ 0 |
| `certifications` | entera | ≥ 0 |
| `hackathons` | entera | ≥ 0 |
| `open_source_contributions` | entera | ≥ 0 |
| `extracurriculars` | entera | ≥ 0 |

### Tratamiento de la entrada
| Caso | Comportamiento |
|---|---|
| Celda vacía | se imputa con la mediana (numéricas) o la moda (categóricas) aprendidas en el entrenamiento, con aviso |
| Columna adicional (incluida `placement_status`) | se ignora, con aviso |
| Categoría no vista en el entrenamiento | se codifica como ceros en el One-Hot, con aviso |
| Columna obligatoria ausente, valor no numérico, fuera de rango o conteo no entero | error: se rechaza todo el archivo, código de salida 1 y no se genera el CSV de salida |
| Modelo inexistente o dañado, archivo de entrada inexistente o vacío | error con mensaje claro y código de salida 1 |

Ninguna fila se elimina en silencio: o se predicen todas o no se genera la salida.

### Salida
Una fila por cada fila de entrada, en el mismo orden:

| Columna | Significado |
|---|---|
| `fila` | posición de la fila en el CSV de entrada (0 = primera fila de datos) |
| `prediccion` | 1 = contratado, 0 = no contratado |
| `etiqueta` | `contratado` o `no contratado` |
| `puntaje_contratado` | puntaje de la clase 1. Por usar `class_weight='balanced'` sirve para ordenar estudiantes, **no es una probabilidad calibrada** |

La predicción usa el umbral 0,5.

## Docker
La imagen usa `python:3.13-slim`, instala `requirements.txt` y copia `src/`, `train.py`, `predict.py` y las pruebas. No tiene `ENTRYPOINT`: sin argumentos muestra la ayuda de `predict.py`. No entrena durante la construcción. Los datos, modelos y resultados se montan desde el equipo, así que persisten aunque el contenedor se elimine (`--rm`). Como la imagen no incluye `.git`, el hash del commit se pasa con `GIT_COMMIT`. El `.dockerignore` excluye `.git`, entornos virtuales, cachés, credenciales y las carpetas `data/`, `models/` y `outputs/`.

Las carpetas de versión y los CSV de predicción se crean con permisos de lectura para todos (755 y 644), así que se pueden abrir desde el equipo aunque el contenedor los cree como root. En Linux, para que además pertenezcan a tu usuario, agrega `--user "$(id -u):$(id -g)"` a `docker run`.

**PowerShell** (desde `fase-2`):
```powershell
docker build -t contratacion-fase2:1.0 .
New-Item -ItemType Directory -Force models, outputs

docker run --rm -e GIT_COMMIT=$(git rev-parse HEAD) `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models" `
  contratacion-fase2:1.0 python train.py `
  --data data/student_placement_synthetic.csv --target placement_status `
  --output-dir models --version v-docker --seed 42

docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/outputs:/app/outputs" `
  contratacion-fase2:1.0 python predict.py `
  --model models/v-docker/modelo.joblib `
  --input data/ejemplo_prediccion.csv `
  --output outputs/predicciones_docker.csv
```

**Bash** (verificado en macOS con zsh y Docker 28.5.1). En Git Bash para Windows hay que anteponer `MSYS_NO_PATHCONV=1` a cada `docker run` para que no se conviertan las rutas `/app/...`.
```bash
docker build -t contratacion-fase2:1.0 .
mkdir -p models outputs
docker run --rm -e GIT_COMMIT=$(git rev-parse HEAD) \
  -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" \
  contratacion-fase2:1.0 python train.py \
  --data data/student_placement_synthetic.csv --target placement_status \
  --output-dir models --version v-docker --seed 42
docker run --rm \
  -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" \
  contratacion-fase2:1.0 python predict.py \
  --model models/v-docker/modelo.joblib \
  --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

## Pruebas
Desde `fase-2`, con el entorno activado (tardan unos segundos):
```bash
python -m pytest
```
También se pueden ejecutar dentro de la imagen: `docker run --rm contratacion-fase2:1.0 python -m pytest`.

Las pruebas generan 300 estudiantes sintéticos con el esquema real (`tests/conftest.py`), entrenan una sola versión para todas las pruebas y escriben en carpetas temporales (`tmp_path`). No descargan nada ni modifican `models/` ni `outputs/`.

| Prueba (`tests/test_aplicacion.py`) | Qué comprueba |
|---|---|
| `test_entrenamiento_genera_artefacto_metricas_y_metadatos` | Se crean `modelo.joblib`, `metrics.json` y `metadata.json`. El artefacto es el pipeline completo con las 16 variables (sin objetivo ni salario), el modelo supera al modelo base y los metadatos traen versión, objetivo, algoritmo, semilla, datos, separación y commit |
| `test_prediccion_conserva_los_resultados_del_modelo_entrenado` | El modelo cargado da una predicción por fila, en orden, con los mismos puntajes del modelo en memoria (tolerancia 1e-6), y predecir no modifica la versión |
| `test_orden_de_columnas_y_columnas_adicionales_no_cambian_la_prediccion` | Reordenar columnas o agregar columnas extra (incluida `placement_status`) da el mismo resultado y se avisa |
| `test_valores_faltantes_se_imputan_con_lo_aprendido_en_entrenamiento` | Una celda vacía (numérica o categórica) da lo mismo que la mediana o la moda guardadas en `metadata.json` |
| `test_celdas_vacias_en_el_csv_se_procesan_desde_la_linea_de_comandos` | `predict.py` acepta un CSV con celdas vacías y predice todas las filas |
| `test_columna_ausente_termina_con_error_y_sin_archivo_de_salida` | Sin `cgpa`, `predict.py` termina con código distinto de 0, explica el error y no crea el CSV de salida |
| `test_tipos_y_rangos_invalidos_se_rechazan` (4 casos) | Texto en una numérica, `cgpa` > 10, `aptitude_score` < 0 y conteo no entero producen `ErrorDatos` |
| `test_categoria_no_vista_se_predice_y_se_informa` | Una carrera desconocida no detiene la predicción y se avisa |
| `test_modelo_inexistente_termina_con_error` | Una versión que no existe termina con código distinto de 0 y sin salida |
| `test_nueva_version_conserva_la_anterior_y_no_se_sobrescribe` | Crear v2 no cambia los archivos de v1; repetir v1 falla (función y `train.py`); ambas versiones se cargan con sus parámetros; no quedan carpetas temporales |
| `test_version_y_predicciones_son_legibles_fuera_del_contenedor` | Las carpetas de versión quedan con permisos 755 y las predicciones con 644 (se omite en Windows) |
| `test_ayuda_disponible` (2 casos) | `train.py --help` y `predict.py --help` funcionan |

## Resultados
Evaluación en el conjunto de prueba (20.000 estudiantes), con `--seed 42`:

| Modelo | ROC-AUC | F1 clase 0 | Recall clase 0 | Precisión clase 0 | Accuracy |
|---|---|---|---|---|---|
| Modelo base (Dummy) | 0,5000 | 0,0000 | 0,0000 | 0,0000 | 0,6847 |
| v1 (`C=1.0`) | **0,6786** | **0,5202** | **0,6450** | **0,4359** | 0,6249 |
| v2 (`C=0.1`) | **0,6786** | **0,5203** | **0,6450** | **0,4360** | 0,6250 |

- **Validación cruzada** (5 partes, solo entrenamiento): ROC-AUC 0,6799 ± 0,0049 en ambas versiones. Coincide con la Fase 1.
- **Frente al modelo base:** el modelo mejora el ROC-AUC y detecta el 64,5 % de los no contratados, que el modelo base nunca detecta. Su accuracy es menor porque el modelo base acierta prediciendo siempre "contratado" (68 % de los casos).
- **v1 frente a v2:** v2 aumenta la regularización y el resultado prácticamente no cambia. Con 80.000 observaciones y 24 variables, el modelo no sobreajusta y la regularización tiene poco efecto.

## Limitaciones
- **Datos sintéticos:** los resultados describen asociaciones en este dataset, no efectos causales del mercado laboral. Antes de usarlos como evidencia habría que validarlos con datos reales de egresados, describiendo su población y periodo. La estabilidad entre la validación y la prueba muestra consistencia en esta distribución, no generalización demostrada al mercado real.
- **Nulos simulados:** el 5 % de nulos por predictora afecta a ≈ 56 % de las filas, más de lo habitual en datos reales.
- **Puntajes no calibrados:** por `class_weight='balanced'`, `puntaje_contratado` no es la probabilidad real de contratación.
- **Umbral sin ajustar:** el umbral 0,5 no se ajustó al costo de cada tipo de error. Si se cambia, debe elegirse solo con validación, nunca con el conjunto de prueba.
- **Reproducibilidad:** la semilla (42) fija la separación, los nulos simulados, la validación cruzada y el modelo, y el optimizador `lbfgs` es determinista. Con los mismos datos y versiones de `requirements.txt` los resultados se repiten; entre sistemas operativos pueden aparecer diferencias numéricas mínimas. En la práctica, Windows, macOS y Docker (Linux) dieron exactamente las mismas métricas.
- **Advertencias en macOS:** en Mac con Apple Silicon, numpy usa la librería Accelerate y emite falsas alarmas `RuntimeWarning: ... encountered in matmul` al entrenar. Como las métricas son idénticas a las de los otros sistemas, `train.py` y `pytest.ini` silencian solo esas advertencias.

## Evidencia de ejecución
**Entrenamiento de v1:**
```
> python train.py --data data/student_placement_synthetic.csv --target placement_status --output-dir models --version v1 --seed 42
Entrenando versión 'v1' con 'data/student_placement_synthetic.csv'...

Versión 'v1' entrenada y guardada.
Evaluación en prueba (20000 filas):
  Métrica             Modelo      Base
  roc_auc             0.6786    0.5000
  f1_clase0           0.5202    0.0000
  recall_clase0       0.6450    0.0000
  precision_clase0    0.4359    0.0000
  accuracy            0.6249    0.6847
  ROC-AUC en validación cruzada (5 partes): 0.6799 ± 0.0049
Archivos generados:
  models\v1\modelo.joblib
  models\v1\metrics.json
  models\v1\metadata.json
```

**Identificador repetido:**
```
> python train.py --data data/student_placement_synthetic.csv --output-dir models --version v1
Error: La versión 'v1' ya existe en 'models\v1'. Elija otro identificador; las versiones existentes no se sobrescriben.
```

**Predicción con el archivo de ejemplo:**
```
> python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
Modelo: models/v1/modelo.joblib (versión: v1)
Filas predichas: 10
  no contratado: 6
  contratado: 4
Aviso: 2 celdas vacías se imputaron con los valores aprendidos en el entrenamiento
Nota: puntaje_contratado ordena a los estudiantes; no es una probabilidad calibrada.
Predicciones guardadas en: outputs\predicciones.csv
```
```
fila,prediccion,etiqueta,puntaje_contratado
0,1,contratado,0.649573
1,1,contratado,0.732018
2,0,no contratado,0.235683
3,1,contratado,0.607217
4,0,no contratado,0.382022
5,0,no contratado,0.421744
6,0,no contratado,0.42219
7,1,contratado,0.558753
8,0,no contratado,0.350546
9,0,no contratado,0.463569
```

**Entrada inválida:**
```
> python predict.py --model models/v1/modelo.joblib --input datos_sin_cgpa.csv --output outputs/x.csv
Error: Faltan columnas obligatorias: cgpa
```
El proceso termina con código 1 y no se crea `outputs/x.csv`.

**Pruebas** (macOS, Python 3.13):
```
$ python -m pytest
============================= test session starts ==============================
collected 16 items

tests/test_aplicacion.py ................                                [100%]

============================== 16 passed in 3.74s ==============================
```
Dentro del contenedor (`docker run --rm contratacion-fase2:1.0 python -m pytest`) también pasan las 16.

**Docker** (macOS con zsh, Docker 28.5.1, comandos de la sección [Docker](#docker) desde `fase-2`):
```
$ docker run --rm -e GIT_COMMIT=$(git rev-parse HEAD) -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" contratacion-fase2:1.0 python train.py --data data/student_placement_synthetic.csv --target placement_status --output-dir models --version v-docker --seed 42
Entrenando versión 'v-docker' con 'data/student_placement_synthetic.csv'...

Versión 'v-docker' entrenada y guardada.
Evaluación en prueba (20000 filas):
  Métrica             Modelo      Base
  roc_auc             0.6786    0.5000
  f1_clase0           0.5202    0.0000
  recall_clase0       0.6450    0.0000
  precision_clase0    0.4359    0.0000
  accuracy            0.6249    0.6847
  ROC-AUC en validación cruzada (5 partes): 0.6799 ± 0.0049
Archivos generados:
  models/v-docker/modelo.joblib
  models/v-docker/metrics.json
  models/v-docker/metadata.json

$ docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" contratacion-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
Modelo: models/v-docker/modelo.joblib (versión: v-docker)
Filas predichas: 10
  no contratado: 6
  contratado: 4
Aviso: 2 celdas vacías se imputaron con los valores aprendidos en el entrenamiento
Nota: puntaje_contratado ordena a los estudiantes; no es una probabilidad calibrada.
Predicciones guardadas en: outputs/predicciones_docker.csv

$ ls -l models/v-docker outputs
models/v-docker:
-rw-r--r--  1 juantabares  staff  6361 Nov  7 20:40 metadata.json
-rw-r--r--  1 juantabares  staff  1228 Nov  7 20:40 metrics.json
-rw-r--r--  1 juantabares  staff  5626 Nov  7 20:40 modelo.joblib

outputs:
-rw-r--r--  1 juantabares  staff  301 Nov  7 20:40 predicciones_docker.csv
```
Los contenedores se eliminan al terminar (`--rm`), pero el modelo y las predicciones quedan en el equipo. Las métricas y predicciones en Docker son idénticas a las de la ejecución local. Una entrada sin `cgpa` dentro del contenedor también termina con `Error: Faltan columnas obligatorias: cgpa` y código 1, y repetir `--version v-docker` termina con código 1 sin modificar la versión existente.

## Fuentes
- Dataset: [Student Placement Prediction Dataset](https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset), Kaggle.
- Notebook de referencia de la autora del dataset: [Student Placement Prediction](https://www.kaggle.com/code/suhanigupta04/student-placement-prediction), adaptado en la Fase 1.
- Documentación de [scikit-learn](https://scikit-learn.org/1.6/) (`Pipeline`, `ColumnTransformer`, persistencia de modelos) y de [Docker](https://docs.docker.com/).

## Versión entregada
> **Pendiente:** etiqueta `entrega-2` en `main` y el hash de su commit.
