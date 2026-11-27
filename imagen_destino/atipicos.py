"""Detección de textos atípicos (Isolation Forest por fuente + regla de ruido) y análisis de n-gramas."""
from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import Config
from .utils import Bitacora

MIN_DOCS_IF = 30           # fuentes con menos textos no se evalúan con Isolation Forest
MIN_DOCS_RUIDO = 300       # la regla de ruido necesita un corpus suficientemente grande
RATIO_RUIDO = 0.5          # proporción de palabras que aparecen una sola vez en todo el corpus


def detectar(df: pd.DataFrame, Z: np.ndarray, cfg: Config, bit: Bitacora) -> pd.DataFrame:
    n = len(df)
    iso_flag = np.zeros(n, dtype=bool)
    puntaje = np.zeros(n)
    omitidas = []
    for fuente, idx in df.groupby("fuente").indices.items():
        if len(idx) < MIN_DOCS_IF:
            omitidas.append(fuente)
            continue
        iso = IsolationForest(n_estimators=300, contamination=cfg.contaminacion, random_state=cfg.semilla).fit(Z[idx])
        iso_flag[idx] = iso.predict(Z[idx]) == -1
        puntaje[idx] = -iso.score_samples(Z[idx])

    ruido = np.zeros(n, dtype=bool)
    if n >= MIN_DOCS_RUIDO:
        df_term = Counter(t for toks in df.tokens for t in set(toks))
        ratio = np.array([np.mean([df_term[t] == 1 for t in toks]) if toks else 0.0 for toks in df.tokens])
        ruido = (ratio >= RATIO_RUIDO) & (df.n_tok.values >= 5)
    out = df.copy()
    out["atipico"] = iso_flag | ruido
    out["motivo_atipico"] = np.where(iso_flag & ruido, "aislamiento + ruido", np.where(iso_flag, "aislamiento", np.where(ruido, "ruido", "")))
    out["puntaje_atipico"] = puntaje
    bit.info(f"Isolation Forest por fuente (contaminación {cfg.contaminacion:g}) sobre el espacio semántico; "
             f"regla de ruido: ≥ {int(RATIO_RUIDO * 100)} % de las palabras del texto aparecen una sola vez en todo el corpus"
             f"{'' if n >= MIN_DOCS_RUIDO else ' (desactivada: corpus pequeño)'}.")
    if omitidas:
        bit.aviso(f"Fuentes con menos de {MIN_DOCS_IF} textos, sin Isolation Forest: {omitidas}.")
    r = out.groupby("fuente").agg(textos=("texto", "size"), atipicos=("atipico", "sum")).reset_index()
    r["pct_atipicos"] = (r.atipicos / r.textos).round(3)
    r["por_ruido"] = [int(out[(out.fuente == f)].motivo_atipico.str.contains("ruido").sum()) for f in r.fuente]
    bit.tabla(r)
    return out


def _contar(listas, n: int) -> Counter:
    c: Counter = Counter()
    for toks in listas:
        if n == 1:
            c.update(toks)
        else:
            c.update(zip(*[toks[i:] for i in range(n)]))
    return c


def ngramas(atipicos: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Unigramas, bigramas y trigramas (palabras ya sin stopwords) de los atípicos, por fuente y en total."""
    filas = []
    grupos = [("todas", atipicos)] + [(f, g) for f, g in atipicos.groupby("fuente")]
    for nombre, g in grupos:
        if len(g) == 0:
            continue
        for n in (1, 2, 3):
            for gram, frec in _contar(g.tokens, n).most_common(top):
                filas.append({"fuente": nombre, "n": n, "ngrama": gram if n == 1 else " ".join(gram), "frecuencia": int(frec)})
    return pd.DataFrame(filas, columns=["fuente", "n", "ngrama", "frecuencia"])
