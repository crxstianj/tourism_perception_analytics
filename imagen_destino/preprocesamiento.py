"""Limpieza y preprocesamiento: URLs, símbolos, números, minúsculas, tokenización, stopwords y stemming.

La etiqueta de fuente (y el indicador es_dmo) viajan en el mismo DataFrame durante todo el flujo.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import snowballstemmer
import stopwordsiso

from .config import Config, IDIOMAS_CANDIDATOS, STEMMERS
from .entrada import ErrorEntrada
from .utils import Bitacora, quitar_acentos

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)", re.I)
_MENCION = re.compile(r"@\w+")


def normalizar(texto: str) -> str:
    t = str(texto).lower()
    t = _URL.sub(" ", t)
    t = _MENCION.sub(" ", t)
    return unicodedata.normalize("NFC", t)


def tokens_basicos(texto: str) -> List[str]:
    """Letras únicamente (sin números, símbolos ni emojis), en minúsculas; conserva stopwords."""
    return [t for t in _TOKEN.findall(normalizar(texto)) if len(t) >= 2]


def stopwords_idioma(idioma: str, extra: List[str] = ()) -> Set[str]:
    if not stopwordsiso.has_lang(idioma):
        raise ErrorEntrada(f"Idioma '{idioma}' no soportado para stopwords. Prueba con es, en, fr, pt, it, de, nl, etc.")
    sw = set(stopwordsiso.stopwords(idioma))
    sw |= {quitar_acentos(w) for w in sw}
    sw |= {w.strip().lower() for w in extra if w.strip()} | {quitar_acentos(w.strip().lower()) for w in extra if w.strip()}
    return sw


def _stemmer(idioma: str):
    nombre = STEMMERS.get(idioma)
    return snowballstemmer.stemmer(nombre) if nombre else None


def _perfil_idioma(tokens: List[str], conjuntos: Dict[str, Set[str]]) -> Dict[str, float]:
    n = max(len(tokens), 1)
    return {l: sum(t in s for t in tokens) / n for l, s in conjuntos.items()}


def procesar(df: pd.DataFrame, cfg: Config, bit: Bitacora):
    """Devuelve (df_incluidos, df_excluidos, mapa_superficie). Agrega columnas de texto procesado."""
    sw = stopwords_idioma(cfg.idioma, cfg.stopwords_extra)
    stem = _stemmer(cfg.idioma)
    if stem is None:
        bit.aviso(f"No hay stemmer Snowball para '{cfg.idioma}': se omite el stemming.")
    cache: Dict[str, str] = {}

    def raiz(w: str) -> str:
        r = cache.get(w)
        if r is None:
            r = stem.stemWord(w) if stem else w
            cache[w] = r
        return r

    cands = list(dict.fromkeys(IDIOMAS_CANDIDATOS + [cfg.idioma]))
    conj = {l: {quitar_acentos(w) for w in stopwordsiso.stopwords(l)} | set(stopwordsiso.stopwords(l))
            for l in cands if stopwordsiso.has_lang(l)}

    basicos, limpios, raices, stems_txt, otro_idioma, idioma_det = [], [], [], [], [], []
    for texto in df.texto.tolist():
        tb = tokens_basicos(texto)
        toks = [t for t in tb if len(t) >= 3 and t not in sw and quitar_acentos(t) not in sw]
        basicos.append(tb)
        limpios.append(toks)
        raices.append([raiz(t) for t in toks])
        # detección de idioma por perfil de stopwords (solo si el texto tiene suficientes palabras)
        det, flag = cfg.idioma, False
        if cfg.filtrar_idioma and len(tb) >= 5:
            perfil = _perfil_idioma(tb, conj)
            mejor = max(perfil, key=perfil.get)
            objetivo = perfil.get(cfg.idioma, 0.0)
            if mejor != cfg.idioma and perfil[mejor] >= 0.2 and perfil[mejor] >= 1.5 * objetivo + 0.05:
                det, flag = mejor, True
        idioma_det.append(det)
        otro_idioma.append(flag)

    out = df.copy()
    out["tokens_basicos"] = basicos
    out["tokens"] = limpios
    out["stems"] = raices
    out["texto_limpio"] = [" ".join(t) for t in limpios]
    out["stems_txt"] = [" ".join(t) for t in raices]
    out["n_tok"] = [len(t) for t in limpios]
    out["idioma_detectado"] = idioma_det

    motivo = np.where(np.array(otro_idioma), "otro idioma", np.where(out.n_tok.values < cfg.min_tokens, "muy corto", ""))
    out["motivo_exclusion"] = motivo
    excl = out[out.motivo_exclusion != ""].copy()
    inc = out[out.motivo_exclusion == ""].copy().reset_index(drop=True)

    resumen = (out.groupby("fuente").agg(registros=("texto", "size"))
               .join(excl.groupby("fuente").motivo_exclusion.value_counts().unstack(fill_value=0))
               .join(inc.groupby("fuente").size().rename("analizados")).fillna(0).astype(int).reset_index())
    bit.info(f"Limpieza: minúsculas, sin URLs/menciones/números/símbolos/emojis; stopwords ({cfg.idioma}) y stemming "
             f"{'Snowball' if stem else '(omitido)'}; tokens de menos de 3 letras descartados.")
    bit.info(f"Se excluyen textos con menos de {cfg.min_tokens} palabras útiles y textos en otro idioma "
             f"({'filtro activo' if cfg.filtrar_idioma else 'filtro desactivado'}).")
    bit.tabla(resumen)
    if inc.empty:
        raise ErrorEntrada("Después de limpiar no quedó ningún texto. Revisa el idioma (--idioma) y el CSV.")
    faltantes = set(df.fuente) - set(inc.fuente)
    for f in faltantes:
        bit.aviso(f"La fuente '{f}' se quedó sin textos analizables (probablemente está en otro idioma que '{cfg.idioma}').")
    if not inc.es_dmo.any():
        raise ErrorEntrada("La DMO se quedó sin textos analizables tras la limpieza (revisa --idioma).")
    if inc.es_dmo.all():
        raise ErrorEntrada("Todas las fuentes percibidas se quedaron sin textos analizables tras la limpieza (revisa --idioma).")

    sup: Dict[str, Counter] = defaultdict(Counter)
    for tk, st in zip(inc.tokens, inc.stems):
        for a, b in zip(tk, st):
            sup[b][a] += 1
    mapa = {s: c.most_common(1)[0][0] for s, c in sup.items()}
    return inc, excl, mapa
