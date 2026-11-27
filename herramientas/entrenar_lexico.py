#!/usr/bin/env python3
"""Entrena un léxico de polaridad desde cualquier CSV con textos y calificaciones (1-5).

Sirve para crear el léxico en español incluido en el repo o para otros idiomas (se usa con --lexico-sentimiento).

  python herramientas/entrenar_lexico.py -i datos/resenas.csv -c texto -r calificacion -o mi_lexico.csv
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from imagen_destino.sentimiento import entrenar_lexico  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--csv", required=True)
    ap.add_argument("-c", "--col-texto", required=True)
    ap.add_argument("-r", "--col-calificacion", required=True)
    ap.add_argument("-o", "--salida", required=True)
    ap.add_argument("--min-df", type=int, default=5, help="apariciones mínimas de una palabra (default 5)")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d[a.col_calificacion] = pd.to_numeric(d[a.col_calificacion], errors="coerce")
    lex, met = entrenar_lexico(d[a.col_texto].astype(str), d[a.col_calificacion], a.semilla, a.min_df)
    lex.to_csv(a.salida, index=False)
    ruta_met = os.path.splitext(a.salida)[0] + "_metricas.json"
    json.dump(met, open(ruta_met, "w"), indent=2)
    print(json.dumps(met, indent=2))
    print(f"Léxico guardado en {a.salida} y métricas en {ruta_met}")


if __name__ == "__main__":
    main()
