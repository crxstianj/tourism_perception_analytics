"""Escritura de resultados (CSV, JSON y bitácora de método en Markdown)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def guardar_csv(df: pd.DataFrame, ruta: str):
    df.to_csv(ruta, index=False, encoding="utf-8-sig")


def guardar_json(obj, ruta: str):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json)


def escribir_bitacora(bit, cfg, ruta: str):
    cab = [f"# Bitácora de método — {cfg.titulo}", "",
           f"Archivo: `{os.path.basename(cfg.csv)}` · DMO: `{cfg.etiqueta_dmo}` · idioma: `{cfg.idioma}` · semilla: {cfg.semilla}", ""]
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(cab) + bit.texto() + "\n")
