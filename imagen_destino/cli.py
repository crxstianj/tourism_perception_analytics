"""Línea de comandos y orquestación del análisis."""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from . import __version__
from .config import Config, PALETAS_ACCESIBLES, PALETAS_OTRAS, IDIOMAS_CANDIDATOS
from .entrada import ErrorEntrada
from .utils import Bitacora

TOTAL = 8


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analizar.py", description="Compara la imagen proyectada por una DMO con la imagen percibida en otras fuentes (UGC).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplo:\n  python analizar.py -i datos/ejemplo.csv -c texto -f fuente -d oficial -l es -t \"Imagen de mi destino\" -p viridis")
    r = p.add_argument_group("parámetros obligatorios")
    r.add_argument("-i", "--csv", required=True, help="ruta del CSV con todos los textos")
    r.add_argument("-c", "--col-texto", required=True, help="columna con el texto")
    r.add_argument("-f", "--col-fuente", required=True, help="columna que identifica la fuente de cada texto")
    r.add_argument("-d", "--dmo", required=True, help="valor de la columna de fuente que corresponde a la DMO (imagen proyectada)")
    r.add_argument("-l", "--idioma", required=True, choices=IDIOMAS_CANDIDATOS, help="idioma principal de los textos")
    r.add_argument("-t", "--titulo", required=True, help="título del tablero")
    r.add_argument("-p", "--paleta", required=True, choices=PALETAS_ACCESIBLES + PALETAS_OTRAS,
                   help="paleta de colores (accesibles: " + ", ".join(PALETAS_ACCESIBLES) + ")")
    o = p.add_argument_group("opcionales")
    o.add_argument("-o", "--salida", default="salida", help="carpeta de resultados (default: salida)")
    o.add_argument("--semilla", type=int, default=42)
    o.add_argument("--col-calificacion", help="columna con calificación 1-5 (solo para medir el acuerdo del sentimiento)")
    o.add_argument("--contaminacion", type=float, default=0.05, help="proporción esperada de atípicos por fuente (default 0.05)")
    o.add_argument("--umbral-particion", type=int, default=300, help="mínimo de textos para modelar tópicos; debajo se usan frecuencias (default 300)")
    o.add_argument("--k-min", type=int, default=3)
    o.add_argument("--k-max", type=int, default=10)
    o.add_argument("--max-puntos", type=int, default=3500, help="máximo de puntos en el mapa (default 3500)")
    o.add_argument("--min-tokens", type=int, default=3, help="palabras útiles mínimas por texto (default 3)")
    o.add_argument("--stopwords-extra", default="", help="palabras a ignorar separadas por coma (p. ej. nombre del destino)")
    o.add_argument("--lexico-sentimiento", help="CSV palabra,peso para idiomas sin sentimiento incluido (ver herramientas/entrenar_lexico.py)")
    o.add_argument("--umbral-polaridad", type=float, default=0.0)
    o.add_argument("--modelo-embeddings", help="ruta LOCAL de un modelo sentence-transformers (opcional; por defecto LSA)")
    o.add_argument("--sin-filtro-idioma", action="store_true", help="no excluir textos detectados en otro idioma")
    o.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def config_desde_args(a) -> Config:
    return Config(csv=a.csv, col_texto=a.col_texto, col_fuente=a.col_fuente, etiqueta_dmo=a.dmo, idioma=a.idioma, titulo=a.titulo,
                  paleta=a.paleta, salida=a.salida, semilla=a.semilla, contaminacion=a.contaminacion, umbral_particion=a.umbral_particion,
                  k_min=a.k_min, k_max=a.k_max, max_puntos=a.max_puntos, min_tokens=a.min_tokens,
                  stopwords_extra=[w.strip() for w in a.stopwords_extra.split(",") if w.strip()], col_calificacion=a.col_calificacion,
                  lexico_sentimiento=a.lexico_sentimiento, umbral_polaridad=a.umbral_polaridad, modelo_embeddings=a.modelo_embeddings,
                  filtrar_idioma=not a.sin_filtro_idioma)


def ejecutar(cfg: Config, bit: Bitacora) -> dict:
    from . import (entrada, preprocesamiento as pp, representacion as rep, atipicos as at, sentimiento as se,
                   topicos as tp, comparacion as cm, visualizacion as vz, reporte as rp)
    from sklearn.manifold import TSNE
    from .utils import muestra_balanceada

    os.makedirs(cfg.salida, exist_ok=True)
    bit.paso(1, TOTAL, "Carga y validación del CSV")
    df = entrada.cargar(cfg, bit)
    bit.paso(2, TOTAL, "Preprocesamiento")
    inc, excl, mapa = pp.procesar(df, cfg, bit)
    if inc.es_dmo.sum() < 5 or (~inc.es_dmo).sum() < 5:
        raise ErrorEntrada("Después de limpiar quedan menos de 5 textos en la DMO o en lo percibido; no es posible comparar.")
    bit.paso(3, TOTAL, "Espacio semántico")
    E = rep.construir(inc, cfg, bit)
    bit.paso(4, TOTAL, "Textos atípicos")
    inc = at.detectar(inc, E.Z, cfg, bit)
    ng = at.ngramas(inc[inc.atipico])
    bit.paso(5, TOTAL, "Sentimiento")
    inc = se.analizar(inc, cfg, bit)
    acuerdo = se.evaluar_con_calificacion(inc, bit)
    bit.paso(6, TOTAL, "Tópicos por sentimiento (solo textos normales)")
    inc = inc.reset_index(drop=True)
    inc["tema"] = ""
    resultados, dists = {}, {}
    for s in ("positivo", "negativo"):
        ii = np.where((~inc.atipico).values & (inc.sentimiento == s).values)[0]
        part = inc.iloc[ii].reset_index(drop=True)
        r = tp.modelar(part, E.Z[ii], s, cfg, bit, mapa)
        resultados[s] = r
        dists[s] = cm.distribucion_temas(r, part)
        if r["metodo"] == "nmf":
            et = {t["indice"]: t["etiqueta"] for t in r["temas"]}
            inc.loc[inc.index[ii], "tema"] = [et.get(a, "") for a in r["asignacion"]]
        if dists[s].get("similitud") is not None:
            bit.info(f"Similitud de la distribución de temas {s} entre DMO y percibida (1 − JSD): {dists[s]['similitud']:.3f}.")
    bit.paso(7, TOTAL, "Comparación entre lo proyectado y lo percibido")
    t_sent = cm.sentimiento(inc, bit)
    t_at = cm.atipicos(inc, bit)
    t_cen = cm.centroides(inc, E.Z, cfg, bit)
    t_cob = cm.cobertura(inc, E.Z, cfg, bit)
    t_dis = cm.terminos_distintivos(inc, mapa)

    bit.paso(8, TOTAL, "Tablero y archivos de salida")
    fuentes = sorted(inc.fuente.unique())
    dmo_nombre = inc[inc.es_dmo].fuente.iloc[0]
    orden = [dmo_nombre] + [f for f in fuentes if f != dmo_nombre]
    cols = dict(zip(orden, vz.colores(cfg.paleta, len(orden))))
    muestra = muestra_balanceada(inc, max(50, cfg.max_puntos // len(fuentes)), cfg.semilla)
    pos = inc.index.get_indexer(muestra.index)
    Zs = E.Z[pos]
    XY = TSNE(2, perplexity=min(30, max(5, len(pos) // 4)), init="pca", random_state=cfg.semilla, learning_rate="auto").fit_transform(Zs)
    figs = {"mapa": vz.mapa(muestra, XY, cols, cfg), "sentimiento": vz.barras_sentimiento(t_sent, cols),
            "atipicos": vz.barras_atipicos(t_at, cols), "similitud": vz.barras_similitud(t_cen, cols),
            "distintivas": vz.barras_distintivas(t_dis, cols[dmo_nombre], cols[[f for f in orden if f != dmo_nombre][0]])}
    cd, cp = cols[dmo_nombre], cols[orden[1]]
    figs["temas_pos"] = vz.distribucion_temas(resultados["positivo"], dists["positivo"], cd, cp, "Textos positivos")
    if resultados["negativo"]["metodo"] == "nmf":
        figs["temas_neg"] = vz.distribucion_temas(resultados["negativo"], dists["negativo"], cd, cp, "Textos negativos")
    kpis = [("textos analizados", f"{len(inc):,}"), ("fuentes", len(fuentes)), ("textos atípicos", f"{int(inc.atipico.sum()):,}"),
            ("% positivo (DMO)", f"{100 * float(t_sent[t_sent.rol == 'DMO'].pct_positivo.iloc[0]):.1f} %")]
    temas_tabla = [t for s in ("positivo", "negativo") for t in resultados[s]["temas"]]
    with open(os.path.join(cfg.salida, "tablero.html"), "w", encoding="utf-8") as f:
        f.write(vz.tablero(cfg, kpis, figs, temas_tabla))

    S = cfg.salida
    rp.guardar_csv(inc[[c for c in ["fila_original", "fuente", "es_dmo", "texto", "idioma_detectado", "n_tok", "polaridad", "sentimiento",
                                     "sin_senal", "atipico", "motivo_atipico", "puntaje_atipico", "tema"] if c in inc.columns]], os.path.join(S, "textos_clasificados.csv"))
    rp.guardar_csv(inc[inc.atipico][["fila_original", "fuente", "motivo_atipico", "puntaje_atipico", "sentimiento", "texto"]].sort_values("puntaje_atipico", ascending=False),
                   os.path.join(S, "atipicos.csv"))
    rp.guardar_csv(ng, os.path.join(S, "ngramas_atipicos.csv"))
    rp.guardar_csv(excl[[c for c in ["fila_original", "fuente", "motivo_exclusion", "texto"] if c in excl.columns]], os.path.join(S, "excluidos.csv"))
    rp.guardar_csv(t_sent, os.path.join(S, "comparacion_sentimiento.csv"))
    rp.guardar_csv(t_at, os.path.join(S, "comparacion_atipicos.csv"))
    rp.guardar_csv(t_cen, os.path.join(S, "similitud_centroides.csv"))
    rp.guardar_csv(t_cob, os.path.join(S, "vecinos_dmo.csv"))
    rp.guardar_csv(t_dis, os.path.join(S, "terminos_distintivos.csv"))
    filas_t = [{"sentimiento": s, "tema": t["etiqueta"], "textos": t["n"], "palabras": ", ".join(t["palabras"]),
                "por_fuente": str(t["por_fuente"]), "texto_representativo_dmo": (t.get("representativo_dmo") or {}).get("texto", ""),
                "texto_representativo_percibida": (t.get("representativo_percibida") or {}).get("texto", "")}
               for s in resultados for t in resultados[s]["temas"]]
    rp.guardar_csv(pd.DataFrame(filas_t), os.path.join(S, "temas.csv"))
    fr = pd.concat([resultados[s]["frecuencias"].assign(sentimiento=s) for s in resultados if len(resultados[s]["frecuencias"])], ignore_index=True) if any(len(resultados[s]["frecuencias"]) for s in resultados) else pd.DataFrame()
    rp.guardar_csv(fr, os.path.join(S, "frecuencias_terminos.csv"))
    rp.guardar_csv(se.muestra_validacion(inc, 40, cfg.semilla), os.path.join(S, "muestra_validacion_sentimiento.csv"))
    resumen = {"version": __version__, "textos_leidos": int(len(df)), "textos_analizados": int(len(inc)), "excluidos": int(len(excl)),
               "fuentes": fuentes, "dmo": dmo_nombre, "acuerdo_sentimiento_calificacion": acuerdo,
               "topicos": {s: {"metodo": resultados[s]["metodo"], "k": resultados[s]["k"], "n": resultados[s]["n"],
                               "coherencias": resultados[s]["coherencias"], "similitud_distribucion": dists[s].get("similitud")} for s in resultados},
               "sim_max_media_dmo_vecinos": t_cob.attrs.get("sim_max_media"), "segundos": round(bit.segundos(), 1)}
    rp.guardar_json(resumen, os.path.join(S, "resumen.json"))
    rp.escribir_bitacora(bit, cfg, os.path.join(S, "bitacora_metodo.md"))
    bit.info(f"Listo en {bit.segundos():.0f} s. Resultados en '{S}/' (abre tablero.html).")
    return resumen


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    a = construir_parser().parse_args(argv)
    cfg = config_desde_args(a)
    try:
        ejecutar(cfg, Bitacora())
    except ErrorEntrada as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    return 0
