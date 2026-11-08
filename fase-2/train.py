"""Entrena una versión del modelo de contratación estudiantil y la guarda en <output-dir>/<version>/.

Ejemplo (desde fase-2):
    python train.py --data data/student_placement_synthetic.csv --target placement_status --output-dir models --version v1 --seed 42
"""

import argparse
import sys
import warnings

from src import config
from src.data import ErrorDatos
from src.storage import ErrorVersion, version_automatica
from src.training import entrenar


def _proporcion(valor):
    numero = float(valor)
    if not 0 < numero < 1:
        raise argparse.ArgumentTypeError("debe estar entre 0 y 1 (excluidos)")
    return numero


def _tasa(valor):
    numero = float(valor)
    if not 0 <= numero < 1:
        raise argparse.ArgumentTypeError("debe estar entre 0 (incluido) y 1")
    return numero


def _folds(valor):
    numero = int(valor)
    if numero == 1 or numero < 0:
        raise argparse.ArgumentTypeError("use 0 para omitir la validación cruzada o un valor >= 2")
    return numero


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Entrena la Regresión Logística balanceada (Fase 1) y guarda el pipeline "
                    "completo, sus métricas y metadatos en una carpeta de versión.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="ruta del CSV con datos etiquetados")
    parser.add_argument("--target", default=config.TARGET,
                        help="nombre de la variable objetivo (binaria 0/1)")
    parser.add_argument("--output-dir", default="models",
                        help="directorio donde se crean las carpetas de versión")
    parser.add_argument("--version", default=None,
                        help="identificador de la versión (p. ej. v1); si se omite se usa la fecha y hora. "
                             "Una versión existente nunca se sobrescribe")
    parser.add_argument("--seed", type=int, default=config.SEMILLA,
                        help="semilla para la separación, los nulos simulados, la validación cruzada y el modelo")
    parser.add_argument("--test-size", type=_proporcion, default=config.TAMANO_PRUEBA,
                        help="proporción de los datos reservada para prueba")
    parser.add_argument("--cv-folds", type=_folds, default=config.FOLDS_CV,
                        help="partes de la validación cruzada estratificada sobre el entrenamiento (0 = omitir)")
    parser.add_argument("--missing-rate", type=_tasa, default=config.TASA_NULOS_SIMULADOS,
                        help="proporción de celdas que se vuelven nulas para simular faltantes como en la Fase 1 (0 = no simular)")
    parser.add_argument("--C", type=float, default=1.0, dest="C",
                        help="inverso de la regularización de la Regresión Logística")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="iteraciones máximas del optimizador de la Regresión Logística")
    return parser


def imprimir_resumen(resultado):
    m = resultado["metricas"]
    print(f"\nVersión '{resultado['version']}' entrenada y guardada.")
    print(f"Evaluación en prueba ({m['n_prueba']} filas):")
    print(f"  {'Métrica':<16}{'Modelo':>10}{'Base':>10}")
    for nombre in ("roc_auc", "f1_clase0", "recall_clase0", "precision_clase0", "accuracy"):
        print(f"  {nombre:<16}{m['modelo'][nombre]:>10.4f}{m['modelo_base'][nombre]:>10.4f}")
    if m["validacion_cruzada"]:
        auc = m["validacion_cruzada"]["roc_auc"]
        print(f"  ROC-AUC en validación cruzada ({m['validacion_cruzada']['folds']} partes): "
              f"{auc['media']:.4f} ± {auc['desviacion']:.4f}")
    if resultado["columnas_ignoradas"]:
        print(f"Columnas ignoradas: {', '.join(resultado['columnas_ignoradas'])}")
    print("Archivos generados:")
    for nombre in ("modelo", "metricas", "metadatos"):
        print(f"  {resultado['rutas'][nombre]}")


def main(argv=None):
    args = crear_parser().parse_args(argv)
    # En macOS con Apple Silicon, numpy usa Accelerate y emite falsas alarmas "encountered in matmul"
    # durante el ajuste; las métricas son idénticas a las de Windows y Linux. Solo se silencian esas.
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
    version = args.version or version_automatica()
    print(f"Entrenando versión '{version}' con '{args.data}'...")
    try:
        resultado = entrenar(
            ruta_datos=args.data,
            directorio_salida=args.output_dir,
            version=version,
            target=args.target,
            semilla=args.seed,
            tamano_prueba=args.test_size,
            folds_cv=args.cv_folds,
            tasa_nulos=args.missing_rate,
            C=args.C,
            max_iter=args.max_iter,
        )
    except (FileNotFoundError, ErrorDatos, ErrorVersion) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    imprimir_resumen(resultado)
    return 0


if __name__ == "__main__":
    sys.exit(main())
