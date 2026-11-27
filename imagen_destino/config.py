"""Configuración, constantes y paletas de color."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Paletas aptas para daltonismo (secuenciales perceptualmente uniformes y Okabe-Ito categórica)
PALETAS_ACCESIBLES = ["viridis", "cividis", "plasma", "inferno", "magma", "okabe-ito"]
PALETAS_OTRAS = ["tab10"]
PALETAS = PALETAS_ACCESIBLES + PALETAS_OTRAS

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]

# Código de idioma -> nombre del stemmer Snowball (si no hay, no se aplica stemming)
STEMMERS = {
    "ar": "arabic", "da": "danish", "nl": "dutch", "en": "english", "fi": "finnish", "fr": "french",
    "de": "german", "hu": "hungarian", "it": "italian", "no": "norwegian", "pt": "portuguese",
    "ro": "romanian", "ru": "russian", "es": "spanish", "sv": "swedish", "tr": "turkish",
}
# Idiomas candidatos para detectar textos en otro idioma (por perfil de stopwords)
IDIOMAS_CANDIDATOS = ["es", "en", "fr", "pt", "it", "de"]

# Idiomas con análisis de sentimiento incluido (el resto requiere --lexico-sentimiento)
SENTIMIENTO_INCLUIDO = ["es", "en"]

NEGADORES = {
    "es": {"no", "nunca", "jamas", "tampoco", "ni", "sin", "nada"},
    "en": {"not", "no", "never", "neither", "nor", "without", "nothing", "cannot"},
    "fr": {"ne", "pas", "jamais", "sans", "rien", "aucun"},
    "pt": {"nao", "nunca", "jamais", "sem", "nada", "tampouco"},
    "it": {"non", "mai", "senza", "niente", "nulla"},
    "de": {"nicht", "kein", "keine", "nie", "ohne", "nichts"},
}


@dataclass
class Config:
    # --- 7 parámetros obligatorios ---
    csv: str
    col_texto: str
    col_fuente: str
    etiqueta_dmo: str
    idioma: str
    titulo: str
    paleta: str
    # --- opcionales ---
    salida: str = "salida"
    semilla: int = 42
    contaminacion: float = 0.05
    umbral_particion: int = 300
    k_min: int = 3
    k_max: int = 10
    max_docs_ajuste: int = 1500
    max_puntos: int = 3500
    min_tokens: int = 3
    stopwords_extra: List[str] = field(default_factory=list)
    col_calificacion: Optional[str] = None
    lexico_sentimiento: Optional[str] = None
    umbral_polaridad: float = 0.0
    modelo_embeddings: Optional[str] = None
    filtrar_idioma: bool = True
    precio: bool = False
    anclas_precio: Optional[str] = None
    umbral_brecha: float = 5.0   # puntos porcentuales
    componentes: int = 100
