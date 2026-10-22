# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** William Villa
- **Código estudiantil:** 1000304912
- **Usuario de GitHub:** [wavilla1](https://github.com/wavilla1)
- **Fecha:** 22 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | [RESPUESTAS T1](RESPUESTAS.md) |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | [RESPUESTAS T2](RESPUESTAS.md) |
| T3 | Inserción frente a búsqueda | 2 | ✅ | [RESPUESTAS T3](RESPUESTAS.md) |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | [RESPUESTAS T4](RESPUESTAS.md) |
| T5 | El costo de la fuerza bruta | 2 | ✅ | [RESPUESTAS T5](RESPUESTAS.md) |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ✅ | [notebook 02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) · [RESPUESTAS A1](RESPUESTAS.md#a1--ejecuta-la-app-rag-de-punta-a-punta) |
| A2 | Tu propio corpus de documentos | 6 | ✅ | [data/company_documents.json](data/company_documents.json) · [captura](evidencias/A2_corpus.png) |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | [RESPUESTAS A3](RESPUESTAS.md) |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | [captura](evidencias/A4_scores.png) · [RESPUESTAS A4](RESPUESTAS.md) |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ✅ | [notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) · [RESPUESTAS B1](RESPUESTAS.md#b1--ejecuta-el-rag-de-papers-y-evalúa-con-el-ground-truth) |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ✅ | [notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) · [RESPUESTAS B2](RESPUESTAS.md) |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | [RESPUESTAS B3](RESPUESTAS.md) |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | [RESPUESTAS B4](RESPUESTAS.md) |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | [RESPUESTAS B5](RESPUESTAS.md) |
| R1 | Reflexión final con tus datos | 10 | ✅ | [RESPUESTAS R1](RESPUESTAS.md) |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | [notebooks/04_challenges.ipynb](notebooks/04_challenges.ipynb) · [gráfica](evidencias/C1_recall.png) |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ⬜ | |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ⬜ | |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ✅ | [eval/preguntas.csv](eval/preguntas.csv) · [RESPUESTAS C4](RESPUESTAS.md) |
| C5 | MMR frente a similitud pura (reto) | 8 | ✅ | [RESPUESTAS C5](RESPUESTAS.md) |
| C6 | Persistencia del índice en disco (reto) | 8 | ⬜ | |

**Autoevaluación:** 100 / 100 puntos — Núcleo 70/70 (T1–T5, A1–A4, B1–B5, R1) · Retos 30/30 (C1 15 + C4 12 + C5 8 = 35, tope 30).

## Cómo ejecutar mis notebooks

1. Abre cada notebook de `notebooks/` en Google Colab.
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el notebook.
3. Ejecuta las celdas en orden.

> No incluyas tu API key en ningún archivo de este repositorio.

## Decisiones y aprendizajes

- Los experimentos del notebook 03 (B2–B5, C4, C5) y del 02b (A1–A4) están en `notebooks/experimentos_03.py` y `notebooks/experimentos_02b.py`; la última celda de cada notebook los ejecuta y muestra las salidas.
- El `score_threshold` de LangChain no filtró nada en ningún experimento: trabaja sobre un score normalizado, distinto del coseno que muestra `similarity_search_with_score`.

- Monté de punta a punta una app RAG en Colab: embeddings con `nvidia/nemotron-3-embed-1b`, índice Qdrant en memoria y generación con `nvidia/nemotron-3-super-120b-a12b`.
- La interfaz se publicó con Gradio (`share=True`), que genera un enlace público temporal mientras el runtime de Colab sigue activo.
- En el RAG de papers, el modelo generador original no estaba disponible (404) y varios otros estaban retirados (410); encontré uno que funcionaba probando una lista de candidatos.
- Los modelos de razonamiento necesitan más `max_completion_tokens`: con 800 las respuestas salían vacías, con 4000 respondió bien las 5 preguntas y el bonus.
- La API key se maneja con los Secrets de Colab (`NVIDIA_API_KEY`), nunca dentro del código ni del repositorio.
- Con solo 4 documentos cortos, cada documento quedó en un único chunk, así que la recuperación depende de qué tan parecido es el texto de la pregunta al de cada documento.
