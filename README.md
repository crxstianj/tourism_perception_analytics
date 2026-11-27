# Imagen del destino: proyectada (DMO) vs. percibida

**Autor:** Cristian Jesus Silva Medel

**Correo:** crxstianj@gmail.com

**Institución:** Universidad de Guanajuato

Herramienta que compara la **imagen proyectada** por una DMO con la **imagen percibida** en otras fuentes
(reseñas, redes sociales, foros) a partir de un CSV. Genera un tablero interactivo `tablero.html` y tablas de resultados.
Funciona **sin conexión a internet** y no depende de ningún destino ni idioma concreto.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Probado con Python 3.13. No descarga modelos ni diccionarios al ejecutarse.

## Uso

```bash
python analizar.py -i datos/ejemplo.csv -c texto -f fuente -d sectur -l es -t "Imagen del destino" -p viridis
```

| Parámetro | Obligatorio | Descripción |
|---|---|---|
| `-i, --csv` | sí | CSV con todos los textos (utf-8, o latin-1; separador `,` `;` o tab) |
| `-c, --col-texto` | sí | columna con el texto |
| `-f, --col-fuente` | sí | columna que indica la fuente de cada texto |
| `-d, --dmo` | sí | valor de esa columna que corresponde a la DMO (lo proyectado); el resto es lo percibido |
| `-l, --idioma` | sí | `es en fr pt it de` |
| `-t, --titulo` | sí | título del tablero |
| `-p, --paleta` | sí | `viridis cividis plasma inferno magma okabe-ito` (accesibles) o `tab10` |
| `-o, --salida` | no | carpeta de resultados (default `salida`) |
| `--col-calificacion` | no | columna con estrellas 1–5, solo para medir el acuerdo del sentimiento |
| `--stopwords-extra` | no | palabras a ignorar (p. ej. el nombre del destino), separadas por coma |
| `--contaminacion` | no | proporción esperada de atípicos por fuente (0.05) |
| `--umbral-particion` | no | mínimo de textos para modelar tópicos; por debajo se usan frecuencias (300) |
| `--k-min / --k-max` | no | rango de tópicos a probar (3–10) |
| `--max-puntos` | no | puntos del mapa (3500) |
| `--lexico-sentimiento` | no | léxico `palabra,peso` para idiomas sin sentimiento incluido |
| `--modelo-embeddings` | no | ruta **local** de un modelo sentence-transformers (opcional) |
| `--sin-filtro-idioma` | no | no excluir textos detectados en otro idioma |

Los errores de entrada (columna inexistente, etiqueta de DMO que no aparece, sin fuentes percibidas, idioma sin sentimiento) se
reportan con un mensaje claro y código de salida 2.

## Qué produce (`salida/`)

`tablero.html` (interactivo; no muestra los métodos), `bitacora_metodo.md` (qué se hizo en cada paso, con cifras),
`textos_clasificados.csv`, `atipicos.csv`, `ngramas_atipicos.csv`, `temas.csv`, `comparacion_sentimiento.csv`,
`comparacion_atipicos.csv`, `similitud_centroides.csv`, `vecinos_dmo.csv`, `terminos_distintivos.csv`,
`frecuencias_terminos.csv`, `excluidos.csv`, `muestra_validacion_sentimiento.csv`, `resumen.json`.

## Cómo está organizado

| Módulo | Función |
|---|---|
| `entrada.py` | lectura y validación del CSV, marca la DMO |
| `preprocesamiento.py` | limpieza, stopwords, stemming, filtro de idioma y de textos muy cortos |
| `representacion.py` | espacio semántico: TF-IDF + SVD (LSA); opcional embeddings locales |
| `atipicos.py` | Isolation Forest por fuente + regla de ruido; n-gramas de los atípicos |
| `sentimiento.py` | positivo/negativo: léxico en español incluido, VADER en inglés, léxico propio en otros idiomas |
| `topicos.py` | NMF por sentimiento con k elegido por coherencia NPMI; frecuencias si hay pocos textos |
| `comparacion.py` | pruebas z con Holm, IC de Wilson, coseno entre centroides con bootstrap, vecinos, 1−JSD, log-odds |
| `visualizacion.py`, `reporte.py` | tablero y archivos |

Los tópicos se calculan solo con textos normales (sin atípicos), separados por sentimiento. No se clasifican categorías turísticas
(gastronomía, cultura…): los temas son agrupaciones de palabras que la persona analista interpreta.

## Limitaciones (léase antes de citar resultados)

- **Sentimiento en español**: el léxico incluido se aprendió de reseñas de TripAdvisor con estrellas. En un 20 % de prueba no usado
  para entrenar: exactitud 0.977, F1 positivo 0.988, **F1 negativo 0.61** (los negativos son pocos y difíciles). Con `--col-calificacion`
  sobre esos mismos datos el acuerdo sale más alto porque son datos de entrenamiento; no es una medida independiente.
  Sobre textos que no son reseñas (publicaciones oficiales, Instagram, Reddit) **no está validado**: use
  `muestra_validacion_sentimiento.csv` para etiquetar a mano una muestra y reportar su propia exactitud.
- **Un solo idioma por ejecución**: los textos en otro idioma se excluyen (aparecen en `excluidos.csv`). No se comparan textos de
  distinto idioma entre sí.
- **Espacio semántico**: LSA es léxico (no entiende sinónimos como un modelo de lenguaje). `--modelo-embeddings` permite usar un
  modelo local.
- Fuentes muy desiguales en tamaño: las pruebas de proporciones son muy sensibles con muchos textos; interprete la magnitud, no solo p.
- El tablero se verificó que genera HTML y JavaScript válidos.

## Herramientas

- `herramientas/unificar_bases.py`: ejemplo de cómo se unieron los 4 CSV originales del curso en un solo archivo.
- `herramientas/entrenar_lexico.py`: entrena un léxico de polaridad para otro idioma/dominio desde un CSV con calificaciones.

## Datos de ejemplo y pruebas

`datos/ejemplo.csv` (1,050 textos, anonimizados: sin URLs, correos, teléfonos ni menciones) muestra el formato; la fuente `sectur`
se usa ahí solo para demostrar el parámetro `-d`. `python -m pytest tests` ejecuta las pruebas.
