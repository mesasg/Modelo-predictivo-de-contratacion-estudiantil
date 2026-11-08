"""Genera predicciones con una versión guardada del modelo, sin reentrenarlo.

Ejemplo (desde fase-2):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""

import argparse
import sys
from pathlib import Path

from src.data import ErrorDatos, leer_csv
from src.prediction import ErrorModelo, cargar_version, guardar_predicciones, predecir


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Carga una versión guardada (pipeline + preprocesamiento) y predice si cada "
                    "estudiante del CSV será contratado (1) o no (0). No reentrena ni reajusta nada.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="archivo modelo.joblib o carpeta de la versión (p. ej. models/v1)")
    parser.add_argument("--input", required=True,
                        help="CSV con las 16 variables de entrada; la variable objetivo no es necesaria")
    parser.add_argument("--output", default="outputs/predicciones.csv",
                        help="CSV de salida con las columnas fila, prediccion, etiqueta y puntaje_contratado")
    return parser


def imprimir_resumen(metadatos, ruta_modelo, resultados, avisos, salida):
    version = metadatos.get("version", "sin metadata.json")
    print(f"Modelo: {ruta_modelo} (versión: {version})")
    print(f"Filas predichas: {len(resultados)}")
    for etiqueta, cantidad in resultados["etiqueta"].value_counts().items():
        print(f"  {etiqueta}: {cantidad}")
    for aviso in avisos:
        print(f"Aviso: {aviso}")
    print("Nota: puntaje_contratado ordena a los estudiantes; no es una probabilidad calibrada.")
    print(f"Predicciones guardadas en: {salida}")


def main(argv=None):
    args = crear_parser().parse_args(argv)
    try:
        if Path(args.output).resolve() == Path(args.input).resolve():
            raise ErrorDatos("--output no puede ser el mismo archivo que --input.")
        modelo, metadatos, avisos = cargar_version(args.model)
        df = leer_csv(args.input)
        resultados, avisos_prediccion = predecir(modelo, df, metadatos)
        salida = guardar_predicciones(resultados, args.output)
    except (OSError, ErrorDatos, ErrorModelo) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    imprimir_resumen(metadatos, args.model, resultados, avisos + avisos_prediccion, salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
