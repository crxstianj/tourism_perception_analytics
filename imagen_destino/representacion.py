"""Espacio vectorial de los documentos (para atípicos, similitud semántica y mapas).

Por defecto: TF-IDF + SVD truncado (análisis semántico latente, LSA), 100 % offline.
Opcional: embeddings de un modelo sentence-transformers que el usuario ya tenga en disco (--modelo-embeddings).
El ajuste se hace sobre una muestra balanceada por fuente para que la fuente más grande no domine el espacio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import Config
from .utils import Bitacora, muestra_balanceada


class Espacio:
    """Guarda el vectorizador y el SVD para proyectar textos nuevos (por ejemplo, los anclajes de precio)."""

    def __init__(self, tfidf, svd, Z, modelo=None):
        self.tfidf, self.svd, self.Z, self.modelo = tfidf, svd, Z, modelo

    def proyectar_stems(self, textos):
        if self.modelo is not None:
            raise RuntimeError("proyectar_stems no aplica con embeddings externos")
        X = self.tfidf.transform(textos)
        return normalize(self.svd.transform(X))


def construir(df: pd.DataFrame, cfg: Config, bit: Bitacora) -> Espacio:
    if cfg.modelo_embeddings:
        try:
            from sentence_transformers import SentenceTransformer  # opcional
        except ImportError:
            bit.aviso("--modelo-embeddings requiere 'sentence-transformers' (no instalado); se usa LSA.")
        else:
            modelo = SentenceTransformer(cfg.modelo_embeddings, local_files_only=True) \
                if _es_ruta_local(cfg.modelo_embeddings) else SentenceTransformer(cfg.modelo_embeddings)
            E = modelo.encode(df.texto.tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            bit.info(f"Embeddings del modelo local '{cfg.modelo_embeddings}' ({E.shape[1]} dimensiones).")
            return Espacio(None, None, np.asarray(E, dtype=np.float32), modelo)

    ajuste = muestra_balanceada(df, cfg.max_docs_ajuste * 2, cfg.semilla)
    min_df = 2 if len(ajuste) > 40 else 1
    tfidf = TfidfVectorizer(min_df=min_df, max_df=0.5, sublinear_tf=True, max_features=30000)
    Xa = tfidf.fit_transform(ajuste.stems_txt)
    k = int(max(2, min(cfg.componentes, Xa.shape[1] - 1, Xa.shape[0] - 1)))
    svd = TruncatedSVD(k, random_state=cfg.semilla).fit(Xa)
    Z = normalize(svd.transform(tfidf.transform(df.stems_txt))).astype(np.float32)
    var = float(svd.explained_variance_ratio_.sum())
    bit.info(f"Espacio semántico: TF-IDF ({Xa.shape[1]:,} términos) + SVD de {k} componentes "
             f"({100 * var:.1f} % de la varianza), ajustado con {len(ajuste):,} textos (tope por fuente: {cfg.max_docs_ajuste * 2:,}).")
    return Espacio(tfidf, svd, Z)


def _es_ruta_local(x: str) -> bool:
    import os
    return os.path.exists(x)
