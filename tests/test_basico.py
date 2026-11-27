import os, sys, subprocess
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imagen_destino.utils import holm, wilson, prueba_dos_proporciones
from imagen_destino.sentimiento import cargar_lexico, puntuar_con_lexico, RECURSOS
from imagen_destino.topicos import coherencia_npmi
from imagen_destino.cli import construir_parser, main

RAIZ = os.path.join(os.path.dirname(__file__), "..")
EJ = os.path.join(RAIZ, "datos", "ejemplo.csv")


def test_holm_monotono():
    p = holm([0.01, 0.04, 0.03])
    assert all(a >= b for a, b in zip(sorted(p), sorted([0.03, 0.06, 0.06])))
    assert max(p) <= 1


def test_wilson_contiene_proporcion():
    lo, hi = wilson(30, 100)
    assert lo < 0.3 < hi


def test_dos_proporciones_iguales():
    dif, p = prueba_dos_proporciones(50, 100, 50, 100)
    assert dif == 0 and p == pytest.approx(1.0)


def test_lexico_negacion():
    pesos, inter = cargar_lexico(os.path.join(RECURSOS, "lexico_polaridad_es.csv"))
    pol, _ = puntuar_con_lexico(["excelente lugar", "pésimo servicio sucio"], pesos, inter, "es")
    assert pol[0] > 0 > pol[1]


def test_parser_exige_siete_parametros():
    with pytest.raises(SystemExit):
        construir_parser().parse_args(["-i", "x.csv"])


def test_error_columna_inexistente(tmp_path, capsys):
    rc = main(["-i", EJ, "-c", "no_existe", "-f", "fuente", "-d", "sectur", "-l", "es", "-t", "t", "-p", "viridis", "-o", str(tmp_path)])
    assert rc == 2


def test_error_dmo_inexistente(tmp_path):
    rc = main(["-i", EJ, "-c", "texto", "-f", "fuente", "-d", "inexistente", "-l", "es", "-t", "t", "-p", "viridis", "-o", str(tmp_path)])
    assert rc == 2


def test_ejecucion_completa(tmp_path):
    rc = main(["-i", EJ, "-c", "texto", "-f", "fuente", "-d", "sectur", "-l", "es", "-t", "Prueba", "-p", "okabe-ito", "-o", str(tmp_path)])
    assert rc == 0
    for f in ["tablero.html", "textos_clasificados.csv", "atipicos.csv", "resumen.json", "bitacora_metodo.md"]:
        assert (tmp_path / f).exists()
    d = pd.read_csv(tmp_path / "textos_clasificados.csv")
    assert set(d.sentimiento) <= {"positivo", "negativo"}
