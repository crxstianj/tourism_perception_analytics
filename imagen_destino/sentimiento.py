"""Análisis de sentimiento binario (positivo / negativo) 100 % offline.

* es  -> léxico de polaridad aprendido de reseñas turísticas con estrellas (recursos/lexico_polaridad_es.csv)
         con manejo de negación; se puntúa por ventanas de ~25 palabras y se promedian.
* en  -> VADER (paquete vaderSentiment, incluye su léxico).
* otro idioma -> léxico del usuario (--lexico-sentimiento ruta.csv con columnas palabra,peso). Se puede crear con
         herramientas/entrenar_lexico.py a partir de cualquier CSV con calificaciones.
Todas las vías producen una polaridad en [-1, 1]; positivo si polaridad >= --umbral-polaridad (0 por defecto).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config, NEGADORES, SENTIMIENTO_INCLUIDO
from .entrada import ErrorEntrada
from .preprocesamiento import tokens_basicos
from .utils import Bitacora, quitar_acentos

RECURSOS = os.path.join(os.path.dirname(__file__), "recursos")
VENTANA = 25
FACTOR_NEGACION = -0.7
INTERCEPTO = "__intercepto__"


def cargar_lexico(ruta: str) -> Tuple[Dict[str, float], float]:
    d = pd.read_csv(ruta)
    if not {"palabra", "peso"} <= set(d.columns):
        raise ErrorEntrada(f"El léxico '{ruta}' debe tener las columnas 'palabra' y 'peso'.")
    d["palabra"] = d["palabra"].astype(str).map(lambda w: quitar_acentos(w.strip().lower()))
    inter = float(d.loc[d.palabra == INTERCEPTO, "peso"].iloc[0]) if (d.palabra == INTERCEPTO).any() else 0.0
    d = d[d.palabra != INTERCEPTO]
    return dict(zip(d.palabra, d.peso.astype(float))), inter


def _ventanas(tokens: List[str], w: int = VENTANA) -> List[List[str]]:
    if len(tokens) <= w:
        return [tokens]
    out = [tokens[i:i + w] for i in range(0, len(tokens), w)]
    if len(out) > 1 and len(out[-1]) < w // 2:
        resto = out.pop()
        out[-1] = out[-1] + resto
    return out


def puntuar_tokens(tokens: List[str], pesos: Dict[str, float], intercepto: float, negadores: set) -> Tuple[float, int]:
    """Puntaje medio por ventana (logit) y número de palabras del léxico encontradas."""
    scores, hits = [], 0
    for v in _ventanas(tokens):
        s, restante = intercepto, 0
        for t in v:
            if t in negadores:
                restante = 3
                continue
            w = pesos.get(t)
            if w is not None:
                s += w * (FACTOR_NEGACION if restante > 0 else 1.0)
                hits += 1
            restante = max(restante - 1, 0)
        scores.append(s)
    return float(np.mean(scores)), hits


def _tokens_lexico(texto: str) -> List[str]:
    return [quitar_acentos(t) for t in tokens_basicos(texto)]


def puntuar_con_lexico(textos, pesos, intercepto, idioma) -> Tuple[np.ndarray, np.ndarray]:
    neg = NEGADORES.get(idioma, set())
    pol, sin = np.zeros(len(textos)), np.zeros(len(textos), dtype=bool)
    for i, t in enumerate(textos):
        s, h = puntuar_tokens(_tokens_lexico(t), pesos, intercepto, neg)
        pol[i] = np.tanh(s / 2.0)
        sin[i] = h == 0
    return pol, sin


def _vader(textos) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as e:  # pragma: no cover
        raise ErrorEntrada("Falta el paquete 'vaderSentiment' (pip install -r requirements.txt).") from e
    sia = SentimentIntensityAnalyzer()
    pol, sin = np.zeros(len(textos)), np.zeros(len(textos), dtype=bool)
    for i, t in enumerate(textos):
        r = sia.polarity_scores(t)
        pol[i] = r["compound"]
        sin[i] = (r["pos"] + r["neg"]) == 0
    return pol, sin


def analizar(df: pd.DataFrame, cfg: Config, bit: Bitacora) -> pd.DataFrame:
    textos = df.texto.tolist()
    if cfg.lexico_sentimiento:
        pesos, inter = cargar_lexico(cfg.lexico_sentimiento)
        pol, sin = puntuar_con_lexico(textos, pesos, inter, cfg.idioma)
        metodo = f"léxico del usuario ({os.path.basename(cfg.lexico_sentimiento)}, {len(pesos):,} palabras)"
    elif cfg.idioma == "es":
        pesos, inter = cargar_lexico(os.path.join(RECURSOS, "lexico_polaridad_es.csv"))
        pol, sin = puntuar_con_lexico(textos, pesos, inter, "es")
        metodo = f"léxico de polaridad en español incluido ({len(pesos):,} palabras, aprendido de reseñas con estrellas) con negación"
    elif cfg.idioma == "en":
        pol, sin = _vader(textos)
        metodo = "VADER (compound)"
    else:
        raise ErrorEntrada(
            f"No hay análisis de sentimiento incluido para '{cfg.idioma}' (incluidos: {SENTIMIENTO_INCLUIDO}). "
            "Entrena un léxico con herramientas/entrenar_lexico.py y pásalo con --lexico-sentimiento.")
    out = df.copy()
    out["polaridad"] = pol
    out["sentimiento"] = np.where(pol >= cfg.umbral_polaridad, "positivo", "negativo")
    out["sin_senal"] = sin
    bit.info(f"Método: {metodo}. Umbral de polaridad: {cfg.umbral_polaridad:g}.")
    tab = (out.groupby("fuente").agg(textos=("texto", "size"), pct_positivo=("sentimiento", lambda s: (s == "positivo").mean()),
                                    pct_sin_senal=("sin_senal", "mean")).reset_index())
    bit.tabla(tab)
    return out


def evaluar_con_calificacion(df: pd.DataFrame, bit: Bitacora) -> Optional[dict]:
    """Si el CSV trae calificaciones (1-5), mide el acuerdo: 4-5 estrellas = positivo, 1-2 = negativo."""
    if "calificacion" not in df.columns:
        return None
    d = df[df.calificacion.notna() & (df.calificacion != 3)]
    if len(d) < 30 or d.calificacion.ge(4).nunique() < 2:
        return None
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    y = (d.calificacion >= 4).astype(int)
    p = (d.sentimiento == "positivo").astype(int)
    r = {"n": int(len(d)), "exactitud": float(accuracy_score(y, p)), "exactitud_balanceada": float(balanced_accuracy_score(y, p)),
         "f1_positivo": float(f1_score(y, p, pos_label=1)), "f1_negativo": float(f1_score(y, p, pos_label=0))}
    bit.info(f"Acuerdo con las calificaciones del CSV (4-5 = positivo, 1-2 = negativo; n={r['n']:,}): "
             f"exactitud {r['exactitud']:.3f}, exactitud balanceada {r['exactitud_balanceada']:.3f}, "
             f"F1 negativo {r['f1_negativo']:.3f}.")
    return r


def muestra_validacion(df: pd.DataFrame, por_fuente: int, semilla: int) -> pd.DataFrame:
    """Muestra estratificada por fuente y sentimiento para etiquetar a mano y validar el clasificador."""
    partes = []
    for _, g in df.groupby("fuente"):
        for _, h in g.groupby("sentimiento"):
            partes.append(h.sample(min(len(h), max(1, por_fuente // 2)), random_state=semilla))
    m = pd.concat(partes)
    return pd.DataFrame({"fila_original": m.fila_original, "fuente": m.fuente, "texto": m.texto.str.slice(0, 600),
                         "sentimiento_modelo": m.sentimiento, "polaridad": m.polaridad.round(3), "etiqueta_manual": ""})


# ----------------------------- entrenamiento del léxico -----------------------------

def entrenar_lexico(textos, calificaciones, semilla: int = 42, min_df: int = 5, peso_minimo: float = 0.1):
    """Aprende un léxico de polaridad (regresión logística sobre presencia de palabras) a partir de textos con
    calificación 1-5 (4-5 = positivo, 1-2 = negativo; 3 se descarta).

    Partición 60/20/20 (entrenamiento / validación / prueba). En validación se eligen C y el umbral de decisión que
    maximizan el F1 macro; el umbral se incorpora al intercepto. Las métricas se miden en prueba (no vista).
    Devuelve (DataFrame palabra,peso, métricas)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    d = pd.DataFrame({"t": list(textos), "r": list(calificaciones)})
    d = d[d.r.notna() & (d.r != 3)].reset_index(drop=True)
    y = (d.r >= 4).astype(int).values
    toks = [[t for t in _tokens_lexico(x) if len(t) >= 3] for x in d.t]
    idx = np.arange(len(d))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=semilla, stratify=y)
    i_fit, i_val = train_test_split(i_tr, test_size=0.25, random_state=semilla, stratify=y[i_tr])
    cv = CountVectorizer(analyzer=lambda x: x, binary=True, min_df=min_df).fit([toks[i] for i in i_fit])
    X = lambda ii: cv.transform([toks[i] for i in ii])
    mejor = None
    for C in (0.1, 0.3, 1.0):
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X(i_fit), y[i_fit])
        dv = m.decision_function(X(i_val))
        for t in np.arange(-2.0, 3.01, 0.1):
            f = f1_score(y[i_val], (dv >= t).astype(int), average="macro")
            if mejor is None or f > mejor[0]:
                mejor = (f, C, float(t), m)
    f_val, C, umbral, m = mejor
    voc = np.array(cv.get_feature_names_out())
    lex = pd.DataFrame({"palabra": voc, "peso": m.coef_[0]})
    lex = lex[lex.peso.abs() >= peso_minimo].sort_values("peso", ascending=False).reset_index(drop=True)
    inter = float(m.intercept_[0]) - umbral
    lex = pd.concat([pd.DataFrame({"palabra": [INTERCEPTO], "peso": [inter]}), lex], ignore_index=True)
    pesos = dict(zip(lex.palabra, lex.peso))
    pesos.pop(INTERCEPTO)
    pred = np.array([1 if np.tanh(puntuar_tokens(toks[i], pesos, inter, set())[0] / 2) >= 0 else 0 for i in i_te])
    met = {"n_total": int(len(d)), "n_entrenamiento": int(len(i_fit)), "n_validacion": int(len(i_val)), "n_prueba": int(len(i_te)),
           "C": C, "umbral_decision": round(umbral, 2), "palabras": int(len(lex) - 1), "f1_macro_validacion": round(float(f_val), 4),
           "exactitud": float(accuracy_score(y[i_te], pred)), "exactitud_balanceada": float(balanced_accuracy_score(y[i_te], pred)),
           "f1_positivo": float(f1_score(y[i_te], pred, pos_label=1)), "f1_negativo": float(f1_score(y[i_te], pred, pos_label=0)),
           "f1_macro": float(f1_score(y[i_te], pred, average="macro"))}
    return lex, met
