"""Modelado de tópicos por sentimiento (NMF sobre TF-IDF) o, si la partición es pequeña, frecuencia de palabras.

El número de tópicos k se elige con coherencia NPMI sobre la co-ocurrencia de las 10 palabras principales de cada tópico.
El ajuste usa una muestra balanceada por fuente; después todos los textos de la partición se asignan a su tópico dominante.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import Config
from .preprocesamiento import _TOKEN  # noqa: F401  (documenta que los tokens vienen del paso 1)
from .representacion import Espacio
from .utils import Bitacora, muestra_balanceada

TOP_PALABRAS = 10


def coherencia_npmi(B, tops: List[List[int]]) -> float:
    """Coherencia NPMI media sobre pares de palabras (co-ocurrencia a nivel documento)."""
    n = B.shape[0]
    eps = 1e-12
    vals = []
    for top in tops:
        S = B[:, top].toarray().astype(np.float64)
        p = S.mean(0)
        co = (S.T @ S) / n
        s = []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                pij = co[i, j]
                if pij <= 0:
                    s.append(-1.0)
                else:
                    s.append(np.log(pij / (p[i] * p[j] + eps)) / (-np.log(pij + eps)))
        vals.append(np.mean(s))
    return float(np.mean(vals))


def _palabras(vec, mapa, tfidf_vocab):
    return [mapa.get(tfidf_vocab[i], tfidf_vocab[i]) for i in vec]


def frecuencias(part: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Términos más frecuentes (por número de textos que los contienen) en la DMO, en lo percibido y en cada fuente."""
    from collections import Counter
    filas = []
    grupos = [("DMO", part[part.es_dmo]), ("percibida", part[~part.es_dmo])] + [(f, g) for f, g in part.groupby("fuente")]
    for nombre, g in grupos:
        if len(g) == 0:
            continue
        c: Counter = Counter()
        for toks in g.tokens:
            c.update(set(toks))
        for w, k in c.most_common(top):
            filas.append({"grupo": nombre, "termino": w, "textos": int(k), "pct_textos": round(k / len(g), 4)})
    return pd.DataFrame(filas, columns=["grupo", "termino", "textos", "pct_textos"])


def _representativo(idx: np.ndarray, Z: np.ndarray, part: pd.DataFrame):
    """Texto más cercano al centroide de los textos `idx` (preferencia por longitud intermedia)."""
    if len(idx) == 0:
        return None
    c = normalize(Z[idx].mean(0, keepdims=True))[0]
    sim = Z[idx] @ c
    orden = idx[np.argsort(-sim)]
    ntok = part.n_tok.values
    elegido = next((i for i in orden if 8 <= ntok[i] <= 80), orden[0])
    fila = part.iloc[int(elegido)]
    return {"fuente": fila.fuente, "texto": fila.texto, "fila_original": int(fila.fila_original)}


def modelar(part: pd.DataFrame, Z: np.ndarray, signo: str, cfg: Config, bit: Bitacora, mapa: Dict[str, str]) -> dict:
    """`part`: textos normales de un sentimiento; `Z`: sus vectores (misma ordenación)."""
    pref = "P" if signo == "positivo" else "N"
    n = len(part)
    res = {"sentimiento": signo, "n": n, "prefijo": pref, "metodo": "frecuencias", "k": 0, "coherencias": {}, "temas": [],
           "asignacion": np.full(n, -1), "frecuencias": frecuencias(part) if n else pd.DataFrame()}
    if n == 0:
        bit.info(f"Partición {signo}: sin textos.")
        return res
    if n < cfg.umbral_particion:
        bit.info(f"Partición {signo}: {n:,} textos < umbral ({cfg.umbral_particion}) → frecuencia de palabras en lugar de tópicos.")
        return res

    ajuste = muestra_balanceada(part, cfg.max_docs_ajuste, cfg.semilla)
    tfidf = TfidfVectorizer(min_df=max(2, int(0.001 * len(ajuste))), max_df=0.5, sublinear_tf=True, max_features=20000)
    try:
        Xf = tfidf.fit_transform(ajuste.stems_txt)
    except ValueError:
        bit.aviso(f"Partición {signo}: vocabulario insuficiente → frecuencia de palabras.")
        return res
    V = Xf.shape[1]
    k_max = min(cfg.k_max, V - 1, max(2, len(ajuste) // 25))
    k_min = max(2, min(cfg.k_min, k_max))
    if V < 30:
        bit.aviso(f"Partición {signo}: vocabulario demasiado pequeño ({V} términos) → frecuencia de palabras.")
        return res
    vocab = tfidf.get_feature_names_out()
    B = (Xf > 0).astype(np.float32).tocsr()
    modelos, coh = {}, {}
    for k in range(k_min, k_max + 1):
        m = NMF(k, init="nndsvda", random_state=cfg.semilla, max_iter=400).fit(Xf)
        tops = [np.argsort(-c)[:TOP_PALABRAS].tolist() for c in m.components_]
        modelos[k], coh[k] = m, coherencia_npmi(B, tops)
    mejor = max(coh.values())
    k = min(kk for kk, v in coh.items() if v >= mejor - 0.01)   # el k más pequeño casi tan coherente como el mejor
    nmf = modelos[k]
    X = tfidf.transform(part.stems_txt)
    W = nmf.transform(X)
    asign = np.where(W.sum(1) > 0, W.argmax(1), -1)
    res.update({"metodo": "nmf", "k": int(k), "coherencias": {int(a): round(float(b), 4) for a, b in coh.items()},
                "asignacion": asign, "n_ajuste": int(len(ajuste))})

    fuentes = part.fuente.values
    for t in range(k):
        idx = np.where(asign == t)[0]
        pal = _palabras(np.argsort(-nmf.components_[t])[:TOP_PALABRAS], mapa, vocab)
        por_fuente = {f: int((fuentes[idx] == f).sum()) for f in sorted(set(fuentes)) if (fuentes[idx] == f).any()}
        pal_fuente = {}
        for f in por_fuente:
            ii = idx[fuentes[idx] == f]
            media = np.asarray(X[ii].mean(0)).ravel()
            pal_fuente[f] = _palabras(np.argsort(-media)[:6], mapa, vocab)
        es_dmo = part.es_dmo.values[idx]
        res["temas"].append({
            "id": f"{pref}{t + 1}", "indice": t, "palabras": pal, "n": int(len(idx)),
            "etiqueta": f"{pref}{t + 1} · " + ", ".join(pal[:3]), "por_fuente": por_fuente, "palabras_por_fuente": pal_fuente,
            "representativo": _representativo(idx, Z, part),
            "representativo_dmo": _representativo(idx[es_dmo], Z, part), "representativo_percibida": _representativo(idx[~es_dmo], Z, part),
        })
    bit.info(f"Partición {signo}: {n:,} textos → NMF con k={k} (coherencia NPMI {coh[k]:.3f}; ajustado con {len(ajuste):,} textos balanceados por fuente).")
    bit.tabla(pd.DataFrame({"k": list(coh), "coherencia_NPMI": [round(v, 3) for v in coh.values()]}))
    bit.tabla(pd.DataFrame([{"tópico": t["etiqueta"], "textos": t["n"], "fuentes": t["por_fuente"]} for t in res["temas"]]))
    return res
