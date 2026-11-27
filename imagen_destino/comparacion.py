"""Comparación entre la imagen proyectada (DMO) y la percibida (resto de fuentes)."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import normalize

from .config import Config
from .utils import Bitacora, holm, prueba_dos_proporciones, wilson

N_BOOT = 300
TOP_K = 10


def _grupo(df):
    return np.where(df.es_dmo.values, "DMO", "percibida")


def sentimiento(df: pd.DataFrame, bit: Bitacora) -> pd.DataFrame:
    """% positivo por fuente (IC Wilson 95 %) y prueba DMO vs cada fuente percibida (z de dos proporciones + Holm)."""
    dmo = df[df.es_dmo]
    kd, nd = int((dmo.sentimiento == "positivo").sum()), len(dmo)
    filas = []
    for f, g in df.groupby("fuente"):
        k, n = int((g.sentimiento == "positivo").sum()), len(g)
        lo, hi = wilson(k, n)
        dif, p = (np.nan, np.nan) if bool(g.es_dmo.iloc[0]) else prueba_dos_proporciones(kd, nd, k, n)
        filas.append({"fuente": f, "rol": "DMO" if bool(g.es_dmo.iloc[0]) else "percibida", "textos": n, "positivos": k,
                      "pct_positivo": k / n, "ic_inf": lo, "ic_sup": hi, "dif_vs_dmo": -dif if dif == dif else np.nan, "p": p})
    t = pd.DataFrame(filas)
    t["p_holm"] = holm(t.p.tolist())
    bit.tabla(t.round(4))
    return t


def atipicos(df: pd.DataFrame, bit: Bitacora) -> pd.DataFrame:
    filas = []
    dmo = df[df.es_dmo]
    kd, nd = int(dmo.atipico.sum()), len(dmo)
    for f, g in df.groupby("fuente"):
        k, n = int(g.atipico.sum()), len(g)
        dif, p = (np.nan, np.nan) if bool(g.es_dmo.iloc[0]) else prueba_dos_proporciones(kd, nd, k, n)
        lo, hi = wilson(k, n)
        filas.append({"fuente": f, "textos": n, "atipicos": k, "pct_atipicos": k / n, "ic_inf": lo, "ic_sup": hi, "p": p})
    t = pd.DataFrame(filas)
    t["p_holm"] = holm(t.p.tolist())
    return t


def centroides(df: pd.DataFrame, Z: np.ndarray, cfg: Config, bit: Bitacora) -> pd.DataFrame:
    """Coseno entre el centroide de la DMO y el de cada fuente percibida, con IC por bootstrap de textos."""
    rng = np.random.default_rng(cfg.semilla)
    dm = np.where(df.es_dmo.values)[0]
    filas = []
    for f, g in df[~df.es_dmo].groupby("fuente"):
        ix = np.where(df.fuente.values == f)[0]
        c = normalize(Z[dm].mean(0, keepdims=True) )[0] @ normalize(Z[ix].mean(0, keepdims=True))[0]
        bs = []
        for _ in range(N_BOOT):
            a = rng.choice(dm, len(dm)); b = rng.choice(ix, len(ix))
            bs.append(normalize(Z[a].mean(0, keepdims=True))[0] @ normalize(Z[b].mean(0, keepdims=True))[0])
        filas.append({"fuente": f, "coseno": float(c), "ic_inf": float(np.percentile(bs, 2.5)), "ic_sup": float(np.percentile(bs, 97.5))})
    t = pd.DataFrame(filas)
    bit.tabla(t.round(3))
    return t


def cobertura(df: pd.DataFrame, Z: np.ndarray, cfg: Config, bit: Bitacora) -> pd.DataFrame:
    """Para cada texto de la DMO se buscan sus TOP_K vecinos más cercanos entre los textos percibidos.
    Devuelve, por fuente percibida, qué proporción de esos vecinos cae en ella frente a lo esperado por su tamaño."""
    rng = np.random.default_rng(cfg.semilla)
    dm = np.where(df.es_dmo.values)[0]
    pe = np.where(~df.es_dmo.values)[0]
    if len(dm) > 1500:
        dm = rng.choice(dm, 1500, replace=False)
    if len(pe) > 20000:
        pe = rng.choice(pe, 20000, replace=False)
    A, B = normalize(Z[dm]), normalize(Z[pe])
    fuentes_pe = df.fuente.values[pe]
    cont: Counter = Counter()
    maxs = []
    for i in range(0, len(A), 500):
        S = A[i:i + 500] @ B.T
        k = min(TOP_K, S.shape[1])
        nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
        maxs.extend(S.max(1))
        for fila in nn:
            cont.update(fuentes_pe[fila])
    tot = sum(cont.values())
    filas = []
    for f, g in pd.Series(fuentes_pe).value_counts().items():
        filas.append({"fuente": f, "pct_vecinos": cont[f] / tot, "pct_esperado": g / len(pe)})
    t = pd.DataFrame(filas)
    t["razon"] = t.pct_vecinos / t.pct_esperado
    bit.info(f"Vecinos más cercanos (top {TOP_K}) de {len(dm):,} textos de la DMO entre {len(pe):,} percibidos; "
             f"similitud coseno máxima media = {np.mean(maxs):.3f}.")
    bit.tabla(t.round(3))
    t.attrs["sim_max_media"] = float(np.mean(maxs))
    return t


def distribucion_temas(res: dict, part: pd.DataFrame) -> dict:
    """Distribución de tópicos en la DMO y en lo percibido y similitud 1 − JSD."""
    from scipy.spatial.distance import jensenshannon
    if res["metodo"] != "nmf":
        return {}
    k = res["k"]
    a = res["asignacion"]
    d = part.es_dmo.values
    vd = np.bincount(a[d & (a >= 0)], minlength=k).astype(float)
    vp = np.bincount(a[~d & (a >= 0)], minlength=k).astype(float)
    if vd.sum() == 0 or vp.sum() == 0:
        return {"dmo": vd.tolist(), "percibida": vp.tolist(), "similitud": None}
    pd_, pp_ = vd / vd.sum(), vp / vp.sum()
    return {"dmo": pd_.tolist(), "percibida": pp_.tolist(), "similitud": float(1 - jensenshannon(pd_, pp_, base=2)),
            "n_dmo": int(vd.sum()), "n_percibida": int(vp.sum())}


def terminos_distintivos(df: pd.DataFrame, mapa: Dict[str, str], top: int = 20, min_total: int = 5) -> pd.DataFrame:
    """Log-odds con prior informativo (Monroe et al.) entre la DMO y lo percibido sobre raíces; se muestra la forma más común."""
    cd, cp = Counter(), Counter()
    for toks, e in zip(df.stems, df.es_dmo):
        (cd if e else cp).update(toks)
    vocab = [w for w in set(cd) | set(cp) if cd[w] + cp[w] >= min_total]
    if not vocab:
        return pd.DataFrame(columns=["termino", "z", "lado", "dmo", "percibida"])
    nd, npp = sum(cd[w] for w in vocab), sum(cp[w] for w in vocab)
    a0 = 1000.0
    prior = {w: (cd[w] + cp[w]) / (nd + npp) * a0 for w in vocab}
    filas = []
    for w in vocab:
        ld = np.log((cd[w] + prior[w]) / (nd + a0 - cd[w] - prior[w]))
        lp = np.log((cp[w] + prior[w]) / (npp + a0 - cp[w] - prior[w]))
        var = 1 / (cd[w] + prior[w]) + 1 / (cp[w] + prior[w])
        filas.append({"termino": mapa.get(w, w), "z": (ld - lp) / np.sqrt(var), "dmo": cd[w], "percibida": cp[w]})
    t = pd.DataFrame(filas)
    t["lado"] = np.where(t.z > 0, "DMO", "percibida")
    return pd.concat([t.sort_values("z", ascending=False).head(top), t.sort_values("z").head(top)]).reset_index(drop=True)
