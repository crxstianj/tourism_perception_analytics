"""Carga y validación del CSV unificado."""
from __future__ import annotations

import pandas as pd

from .config import Config
from .utils import Bitacora


class ErrorEntrada(Exception):
    """Problema con el archivo o con los parámetros de entrada (mensaje pensado para el usuario)."""


def _leer(ruta: str) -> pd.DataFrame:
    ultimo = None
    for enc in ("utf-8-sig", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(ruta, encoding=enc, sep=sep)
            except UnicodeDecodeError as e:
                ultimo = e
                break  # probar otra codificación
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                ultimo = e
                continue
            except FileNotFoundError:
                raise ErrorEntrada(f"No se encontró el archivo CSV: {ruta}")
            if df.shape[1] > 1 or sep == "\t":
                return df
            ultimo = ValueError("una sola columna")
    raise ErrorEntrada(f"No se pudo leer el CSV ({ruta}): {ultimo}")


def cargar(cfg: Config, bit: Bitacora) -> pd.DataFrame:
    df = _leer(cfg.csv)
    columnas = list(df.columns)
    for nombre, valor in (("--col-texto", cfg.col_texto), ("--col-fuente", cfg.col_fuente)):
        if valor not in columnas:
            raise ErrorEntrada(f"La columna '{valor}' ({nombre}) no existe en el CSV. Columnas disponibles: {columnas}")
    if cfg.col_calificacion and cfg.col_calificacion not in columnas:
        raise ErrorEntrada(f"La columna de calificación '{cfg.col_calificacion}' no existe. Columnas: {columnas}")

    n_bruto = len(df)
    out = pd.DataFrame({
        "fila_original": range(n_bruto),
        "texto": df[cfg.col_texto].astype("string"),
        "fuente": df[cfg.col_fuente].astype("string").str.strip(),
    })
    if cfg.col_calificacion:
        out["calificacion"] = pd.to_numeric(df[cfg.col_calificacion], errors="coerce")

    vacios = out.texto.isna() | (out.texto.str.strip() == "") | out.fuente.isna() | (out.fuente == "")
    out = out[~vacios].copy()
    n_vacios = int(vacios.sum())
    out["texto"] = out.texto.astype(str)
    out["fuente"] = out.fuente.astype(str)
    dup = out.duplicated(["fuente", "texto"])
    n_dup = int(dup.sum())
    out = out[~dup].reset_index(drop=True)

    etiqueta = cfg.etiqueta_dmo.strip().lower()
    out["es_dmo"] = out.fuente.str.lower() == etiqueta
    conteo = out.groupby("fuente").size().sort_values(ascending=False)
    if not out.es_dmo.any():
        raise ErrorEntrada(
            f"No hay registros con la etiqueta de la DMO '{cfg.etiqueta_dmo}' en la columna '{cfg.col_fuente}'. "
            f"Valores encontrados: {conteo.to_dict()}")
    if out.es_dmo.all():
        raise ErrorEntrada("Todos los registros pertenecen a la DMO: se necesita al menos una fuente percibida.")
    # Unificar la etiqueta de la DMO (mayúsculas/espacios) y conservar el resto tal cual
    out.loc[out.es_dmo, "fuente"] = cfg.etiqueta_dmo.strip()

    bit.info(f"Archivo leído: {n_bruto:,} filas; sin texto o sin fuente: {n_vacios:,}; duplicados dentro de la misma fuente: {n_dup:,}.")
    bit.info(f"Etiqueta de la DMO: '{cfg.etiqueta_dmo.strip()}'. Fuentes percibidas: {[f for f in conteo.index if f.lower() != etiqueta]}.")
    bit.tabla(out.groupby(["fuente", "es_dmo"]).size().rename("registros").reset_index()
              .assign(rol=lambda d: d.es_dmo.map({True: "DMO (proyectada)", False: "percibida"}))[["fuente", "rol", "registros"]])
    return out
