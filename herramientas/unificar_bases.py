#!/usr/bin/env python3
"""Unifica las bases originales del proyecto en un solo CSV con columnas: texto, fuente, calificacion.

Uso:
  python herramientas/unificar_bases.py --tripadvisor TripAdvisor.csv --sectur sectur.csv \
      --instagram instaUGC.csv --reddit redditcdmx.csv --salida datos/base_unificada.csv

Cada archivo es opcional; se unifican los que se indiquen. Formatos esperados:
  TripAdvisor: columnas texto, date, target (calificación 1-5)
  Sectur:      columnas clase, texto
  Instagram:   columna description
  Reddit:      columnas title, text, comments (los comentarios vienen separados por " - ")
"""
import argparse
import re

import pandas as pd

BOT = re.compile(r"i am a bot|this action was performed automatically|weekly destination thread|read what redditors", re.I)
SEPARADOR_REDDIT = re.compile(r"\s+-\s+")


def tripadvisor(ruta):
    d = pd.read_csv(ruta)
    d = d[d["target"].astype(str).isin(list("12345"))]      # descarta filas corruptas
    return pd.DataFrame({"texto": d["texto"], "fuente": "tripadvisor", "calificacion": d["target"].astype(int)})


def sectur(ruta):
    d = pd.read_csv(ruta)
    return pd.DataFrame({"texto": d["texto"], "fuente": "sectur", "calificacion": pd.NA})


def instagram(ruta):
    d = pd.read_csv(ruta)
    return pd.DataFrame({"texto": d["description"], "fuente": "instagram", "calificacion": pd.NA})


def reddit(ruta):
    d = pd.read_csv(ruta)
    filas = []
    for _, r in d.iterrows():
        base = f"{r['title']} {r['text'] if isinstance(r['text'], str) else ''}".strip()
        filas.append(base)
        if isinstance(r["comments"], str):
            filas += [c.strip() for c in SEPARADOR_REDDIT.split(r["comments"]) if c.strip()]
    filas = [f for f in filas if not BOT.search(f)]            # mensajes automáticos del subreddit
    return pd.DataFrame({"texto": filas, "fuente": "reddit", "calificacion": pd.NA})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tripadvisor"); ap.add_argument("--sectur"); ap.add_argument("--instagram"); ap.add_argument("--reddit")
    ap.add_argument("--salida", default="datos/base_unificada.csv")
    a = ap.parse_args()
    partes = []
    for nombre, f in (("tripadvisor", tripadvisor), ("sectur", sectur), ("instagram", instagram), ("reddit", reddit)):
        ruta = getattr(a, nombre)
        if ruta:
            p = f(ruta)
            print(f"{nombre}: {len(p):,} registros")
            partes.append(p)
    if not partes:
        ap.error("indica al menos un archivo de entrada")
    out = pd.concat(partes, ignore_index=True)
    out["texto"] = out["texto"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = out[out["texto"] != ""]
    out.to_csv(a.salida, index=False)
    print(f"Guardado {a.salida}: {len(out):,} filas")


if __name__ == "__main__":
    main()
