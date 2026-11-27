"""Tablero interactivo (un solo HTML, plotly.js incluido). No describe los métodos usados: solo muestra resultados."""
from __future__ import annotations

import html
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config, OKABE_ITO

FUENTE_UI = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def colores(paleta: str, n: int) -> List[str]:
    if paleta == "okabe-ito":
        base = [c for c in OKABE_ITO if c.lower() != "#000000"] or OKABE_ITO
        return [base[i % len(base)] for i in range(n)]
    if paleta == "tab10":
        q = pc.qualitative.D3
        return [q[i % len(q)] for i in range(n)]
    pos = [0.08 + 0.78 * i / max(n - 1, 1) for i in range(n)]
    return [c if c.startswith("#") or c.startswith("rgb") else c for c in pc.sample_colorscale(paleta.capitalize(), pos)]


def _ejes(fig, **kw):
    kw.setdefault("margin", dict(l=50, r=20, t=40, b=40))
    fig.update_layout(font=dict(family=FUENTE_UI, size=13), paper_bgcolor="white", plot_bgcolor="white", **kw)


def mapa(df: pd.DataFrame, XY: np.ndarray, col: Dict[str, str], cfg: Config) -> go.Figure:
    fig = go.Figure()
    combos = []
    for f in sorted(df.fuente.unique()):
        for s in ("positivo", "negativo"):
            for a in (False, True):
                m = ((df.fuente == f) & (df.sentimiento == s) & (df.atipico == a)).values
                if not m.any():
                    continue
                d = df[m]
                tx = d.texto.str.slice(0, 700).map(html.escape)
                cd = np.stack([d.fuente, d.sentimiento, d.tema.fillna(""), np.where(d.atipico, "atípico", "normal"), tx], axis=1)
                primero = not any(c[0] == f for c in combos)
                fig.add_trace(go.Scattergl(
                    x=XY[m, 0], y=XY[m, 1], mode="markers", name=f"{f}", legendgroup=f, showlegend=primero, customdata=cd,
                    marker=dict(size=8 if a else 5, color=col[f], opacity=0.95 if a else 0.5, symbol="x" if a else "circle",
                                line=dict(width=1 if a else 0, color=col[f])),
                    hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]} · %{customdata[3]}<extra></extra>"))
                combos.append((f, s, a))

    def vis(fs):
        return [fs(c) for c in combos]
    botones = [
        dict(label="Todo", method="update", args=[{"visible": vis(lambda c: True)}]),
        dict(label="Solo positivos", method="update", args=[{"visible": vis(lambda c: c[1] == "positivo")}]),
        dict(label="Solo negativos", method="update", args=[{"visible": vis(lambda c: c[1] == "negativo")}]),
        dict(label="Solo atípicos", method="update", args=[{"visible": vis(lambda c: c[2])}]),
        dict(label="Solo normales", method="update", args=[{"visible": vis(lambda c: not c[2])}]),
    ]
    _ejes(fig, height=620, legend=dict(orientation="h", y=-0.08, itemsizing="constant"),
          updatemenus=[dict(type="buttons", direction="right", x=0, y=1.1, buttons=botones, showactive=True)])
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def barras_sentimiento(t: pd.DataFrame, col: Dict[str, str]) -> go.Figure:
    t = t.sort_values(["rol", "fuente"], ascending=[True, True])
    fig = go.Figure(go.Bar(
        x=t.fuente, y=t.pct_positivo * 100, marker_color=[col[f] for f in t.fuente],
        error_y=dict(type="data", symmetric=False, array=(t.ic_sup - t.pct_positivo) * 100, arrayminus=(t.pct_positivo - t.ic_inf) * 100),
        customdata=np.stack([t.textos, t.rol], 1), hovertemplate="%{x} (%{customdata[1]})<br>%{y:.1f} % positivo · n=%{customdata[0]}<extra></extra>"))
    _ejes(fig, height=360, yaxis_title="% de textos positivos", showlegend=False)
    fig.update_yaxes(range=[max(0, float((t.ic_inf.min() * 100) - 8)), 100.5])
    return fig


def barras_atipicos(t: pd.DataFrame, col: Dict[str, str]) -> go.Figure:
    fig = go.Figure(go.Bar(x=t.fuente, y=t.pct_atipicos * 100, marker_color=[col[f] for f in t.fuente],
                           customdata=np.stack([t.textos, t.atipicos], 1),
                           hovertemplate="%{x}<br>%{y:.1f} % atípicos (%{customdata[1]} de %{customdata[0]})<extra></extra>"))
    _ejes(fig, height=360, yaxis_title="% de textos atípicos", showlegend=False)
    return fig


def distribucion_temas(res: dict, dist: dict, col_dmo: str, col_per: str, titulo: str) -> go.Figure:
    fig = go.Figure()
    if not dist or not res.get("temas"):
        _ejes(fig, height=100)
        return fig
    et = [t["etiqueta"] for t in res["temas"]]
    fig.add_bar(y=et, x=np.array(dist["dmo"]) * 100, orientation="h", name="Proyectada (DMO)", marker_color=col_dmo)
    fig.add_bar(y=et, x=np.array(dist["percibida"]) * 100, orientation="h", name="Percibida", marker_color=col_per)
    _ejes(fig, height=max(320, 70 * len(et)), barmode="group", xaxis_title="% de los textos de cada lado", title=dict(text=titulo, x=0),
          legend=dict(orientation="h", y=1.12), margin=dict(l=240, r=20, t=70, b=40))
    fig.update_yaxes(autorange="reversed")
    return fig


def barras_distintivas(t: pd.DataFrame, col_dmo: str, col_per: str, top: int = 12) -> go.Figure:
    a = t[t.lado == "DMO"].sort_values("z", ascending=False).head(top)
    b = t[t.lado == "percibida"].sort_values("z").head(top)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Más propios de la imagen proyectada", "Más propios de la imagen percibida"), horizontal_spacing=0.18)
    fig.add_bar(y=a.termino, x=a.z, orientation="h", marker_color=col_dmo, row=1, col=1, showlegend=False,
                customdata=np.stack([a.dmo, a.percibida], 1), hovertemplate="%{y}: DMO %{customdata[0]} · percibida %{customdata[1]}<extra></extra>")
    fig.add_bar(y=b.termino, x=-b.z, orientation="h", marker_color=col_per, row=1, col=2, showlegend=False,
                customdata=np.stack([b.dmo, b.percibida], 1), hovertemplate="%{y}: DMO %{customdata[0]} · percibida %{customdata[1]}<extra></extra>")
    _ejes(fig, height=460)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="distintividad (z)")
    return fig


def barras_similitud(cent: pd.DataFrame, col: Dict[str, str]) -> go.Figure:
    t = cent.sort_values("coseno")
    fig = go.Figure(go.Bar(y=t.fuente, x=t.coseno, orientation="h", marker_color=[col[f] for f in t.fuente],
                           error_x=dict(type="data", symmetric=False, array=t.ic_sup - t.coseno, arrayminus=t.coseno - t.ic_inf),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    _ejes(fig, height=max(240, 90 * len(t)), xaxis_title="similitud con la imagen proyectada (coseno)", showlegend=False)
    fig.update_xaxes(range=[0, 1])
    return fig


_CSS = """body{margin:0;font-family:%s;color:#1f2933;background:#f6f7f9}header{background:#fff;padding:22px 32px;border-bottom:1px solid #e3e6ea}
h1{margin:0;font-size:24px}header p{margin:6px 0 0;color:#5b6670}main{max-width:1180px;margin:0 auto;padding:20px 24px 60px}
section{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:18px 22px;margin:18px 0}h2{margin:0 0 4px;font-size:18px}
.sub{color:#5b6670;font-size:13px;margin:0 0 10px}.kpis{display:flex;gap:14px;flex-wrap:wrap}.kpi{flex:1;min-width:150px;border:1px solid #e3e6ea;border-radius:8px;padding:10px 14px}
.kpi b{display:block;font-size:22px}.kpi span{font-size:12px;color:#5b6670}#detalle{margin-top:10px;padding:12px 14px;border:1px solid #e3e6ea;border-radius:8px;background:#fafbfc;min-height:70px;font-size:14px;white-space:pre-wrap}
.tabla{border-collapse:collapse;width:100%%;font-size:13px}.tabla td,.tabla th{border-bottom:1px solid #e9ecef;padding:6px 8px;text-align:left;vertical-align:top}
""" % FUENTE_UI

_JS_CLICK = """
document.addEventListener('DOMContentLoaded',function(){var g=document.getElementById('mapa').getElementsByClassName('plotly-graph-div')[0];
if(!g)return;g.on('plotly_click',function(e){var c=e.points[0].customdata;
document.getElementById('detalle').innerHTML='<b>'+c[0]+'</b> · '+c[1]+' · '+c[3]+(c[2]?' · tema: '+c[2]:'')+'<br><br>'+c[4];});});
"""


def _fig_html(fig, id_, primera=False):
    return f'<div id="{id_}">' + fig.to_html(full_html=False, include_plotlyjs="inline" if primera else False, config={"displaylogo": False, "responsive": True}) + "</div>"


def tablero(cfg: Config, kpis: List[tuple], figs: Dict[str, go.Figure], temas_tabla: List[dict]) -> str:
    kp = "".join(f'<div class="kpi"><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>' for k, v in kpis)
    filas = "".join(
        f"<tr><td><b>{html.escape(t['etiqueta'])}</b></td><td>{t['n']:,}</td><td>{html.escape(', '.join(t['palabras'][:8]))}</td>"
        f"<td>{html.escape((t.get('representativo') or {}).get('texto', '')[:260])}</td></tr>" for t in temas_tabla)
    secciones = [
        ("Mapa de textos", "Cada punto es un texto; los más cercanos hablan de cosas parecidas. Las × son textos atípicos. Haz clic en un punto para leerlo.",
         _fig_html(figs["mapa"], "mapa", True) + '<div id="detalle">Selecciona un punto del mapa.</div>'),
        ("Tono de los textos", "Porcentaje de textos positivos por fuente, con intervalo de confianza del 95 %.", _fig_html(figs["sentimiento"], "sent")),
        ("Textos atípicos", "Proporción de textos que se alejan de lo habitual en su propia fuente.", _fig_html(figs["atipicos"], "atip")),
        ("Cercanía entre lo proyectado y lo percibido", "Qué tanto se parece el contenido de cada fuente percibida al de la DMO.", _fig_html(figs["similitud"], "sim")),
        ("Temas en textos positivos", "Reparto de temas en lo proyectado frente a lo percibido.", _fig_html(figs["temas_pos"], "tp")),
    ]
    if "temas_neg" in figs:
        secciones.append(("Temas en textos negativos", "Reparto de temas en lo proyectado frente a lo percibido.", _fig_html(figs["temas_neg"], "tn")))
    secciones += [("Palabras que distinguen cada imagen", "Términos relativamente más frecuentes en un lado que en el otro.", _fig_html(figs["distintivas"], "dist")),
                  ("Temas y texto representativo", "Cada tema con sus palabras principales y un texto cercano a su centro.",
                   f'<table class="tabla"><tr><th>Tema</th><th>Textos</th><th>Palabras</th><th>Texto representativo</th></tr>{filas}</table>')]
    cuerpo = "".join(f'<section><h2>{html.escape(t)}</h2><p class="sub">{html.escape(s)}</p>{c}</section>' for t, s, c in secciones)
    return (f'<!doctype html><html lang="{cfg.idioma}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(cfg.titulo)}</title><style>{_CSS}</style></head><body><header><h1>{html.escape(cfg.titulo)}</h1>"
            f"<p>Imagen proyectada ({html.escape(cfg.etiqueta_dmo)}) frente a imagen percibida</p></header><main><section><div class=\"kpis\">{kp}</div></section>"
            f"{cuerpo}</main><script>{_JS_CLICK}</script></body></html>")
