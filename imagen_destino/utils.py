"""Utilidades compartidas: bitácora, muestreo, estadística básica y color."""
from __future__ import annotations

import time
import unicodedata
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def quitar_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


class Bitacora:
    """Registra lo que hace cada paso (para el documento de método) y avisa por consola."""

    def __init__(self, silenciosa: bool = False):
        self.lineas: List[str] = []
        self.silenciosa = silenciosa
        self._t0 = time.time()

    def paso(self, n: int, total: int, titulo: str):
        msg = f"[{n}/{total}] {titulo}"
        if not self.silenciosa:
            print(msg, flush=True)
        self.lineas += ["", f"## {n}. {titulo}", ""]

    def info(self, texto: str):
        if not self.silenciosa:
            print("      " + texto, flush=True)
        self.lineas.append(f"- {texto}")

    def aviso(self, texto: str):
        if not self.silenciosa:
            print("      AVISO: " + texto, flush=True)
        self.lineas.append(f"- **Aviso:** {texto}")

    def tabla(self, df: pd.DataFrame, index: bool = False):
        if df is None or len(df) == 0:
            return
        t = df.reset_index() if index else df
        cab = "| " + " | ".join(str(c) for c in t.columns) + " |"
        sep = "|" + "|".join(["---"] * len(t.columns)) + "|"
        filas = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in t.itertuples(index=False)]
        self.lineas += ["", cab, sep] + filas + [""]

    def texto(self) -> str:
        return "\n".join(self.lineas)

    def segundos(self) -> float:
        return time.time() - self._t0


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v).replace("|", "/").replace("\n", " ")


def muestra_balanceada(df: pd.DataFrame, tope: int, semilla: int, columna: str = "fuente") -> pd.DataFrame:
    """Toma como máximo `tope` filas por fuente para que ninguna fuente domine el ajuste de un modelo."""
    partes = []
    for _, g in df.groupby(columna, sort=True):
        partes.append(g if len(g) <= tope else g.sample(tope, random_state=semilla))
    return pd.concat(partes) if partes else df.iloc[0:0]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def prueba_dos_proporciones(k1: int, n1: int, k2: int, n2: int):
    """Prueba z de dos proporciones (bilateral). Devuelve (diferencia, p)."""
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2, 1.0)
    z = (p1 - p2) / se
    return (p1 - p2, float(2 * stats.norm.sf(abs(z))))


def holm(pvals: List[float]) -> List[float]:
    """Corrección de Holm-Bonferroni (devuelve p ajustados en el orden original)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out.tolist()
    orden = idx[np.argsort(p[idx])]
    m = len(orden)
    previo = 0.0
    for rango, i in enumerate(orden):
        aj = min(1.0, (m - rango) * p[i])
        previo = max(previo, aj)
        out[i] = previo
    return out.tolist()


def pct(x: float, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f} %"
