# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ cálculo a mano + simulación en Python (abajo).

_Mi respuesta / análisis:_

Con M = 16, la probabilidad de que un vector llegue al menos a la capa k es 16⁻ᵏ. Con N = 1.000.000:

| Capa | Teórico (a mano) | Simulado |
|---|---:|---:|
| ≥ 1 | 1.000.000 / 16 = 62.500 | 62.085 |
| ≥ 2 | 1.000.000 / 256 ≈ 3.906 | 3.972 |
| ≥ 3 | 1.000.000 / 4.096 ≈ 244 | 231 |

```python
import numpy as np, math
rng = np.random.default_rng(42); N = 1_000_000; M = 16; mL = 1 / math.log(M)
l = np.floor(-np.log(rng.random(N)) * mL).astype(int)
for k in (1, 2, 3): print(k, N / M**k, (l >= k).sum())   # capa máxima obtenida: 5
```

La simulación coincide con la teoría: cada capa tiene unas 16 veces menos nodos que la de abajo.

Que haya pocos hubs arriba es bueno porque en las capas altas la búsqueda da "saltos largos" entre muy pocos nodos y llega rápido a la región correcta del espacio. Después solo tiene que refinar en las capas bajas, en vez de recorrer el millón de nodos de la capa 0.


## T2 · ef, k y el tamaño de la cola

_Mi respuesta / análisis:_

1. **k = 10 con ef = 5:** la cola de candidatos solo guarda 5 elementos, así que la búsqueda no puede devolver 10 vecinos (o devuelve menos de los pedidos). Por eso ef debe ser siempre ≥ k.
2. **Valor inicial:** conviene empezar con ef igual o un poco mayor que k, y subirlo cuando el recall no es suficiente. Más ef mejora la precisión pero hace cada consulta más lenta.
3. **Cambiable en producción:** **ef** (ef_search), porque solo controla cuántos candidatos se exploran al buscar. **ef_construction** se usó para decidir las conexiones al construir el grafo; cambiarlo implica reconstruir el índice.


## T3 · Inserción frente a búsqueda

_Mi respuesta / análisis:_

| | Inserción | Búsqueda | Por qué |
|---|---|---|---|
| ¿Se sortea la capa? | Sí | No | Solo un vector nuevo necesita una capa máxima; la consulta no se guarda en el grafo, solo lo recorre desde la entrada. |
| Valor de ef en las capas altas | 1 | 1 | En las capas altas solo interesa bajar rápido hacia el vecino más cercano; con 1 candidato basta. |
| Parámetro de la fase principal | ef_construction | ef (ef_search) | Al insertar, ef_construction define cuántos candidatos se evalúan para elegir buenas conexiones; al buscar, ef define cuántos se exploran para encontrar los k vecinos. |
| ¿Modifica el índice? | Sí | No | La inserción agrega el nodo y sus aristas; la búsqueda es solo lectura. |

En el notebook 03, `vectorstore.add_documents(...)` dispara la **inserción** y `retriever.invoke(...)` dispara la **búsqueda**.


## T4 · El invariante de FAISSVectorStore

_Mi respuesta / análisis:_

| Posición | Antes | Después de borrar B (solo en `self.documents`) |
|---|---|---|
| 0 | A | A |
| 1 | B | C |
| 2 | C | D |
| 3 | D | — (no existe) |

FAISS no se enteró del borrado, así que una búsqueda que antes daba [1, 2, 3] (B, C, D) sigue devolviendo los índices [1, 2, 3], pero ahora se traducen a **[C, D, IndexError]**:

- Posiciones 1 y 2 → **error silencioso**: devuelven C y D cuando el vector encontrado era el de B y el de C. La respuesta es equivocada y no hay ninguna excepción.
- Posición 3 → **lanza excepción** (IndexError), porque la lista ya solo tiene 3 elementos.

Mantener sincronizados a mano dos estructuras (índice y lista de documentos) es frágil. Bases vectoriales como Qdrant guardan el vector, el payload y un id en el mismo registro, y permiten borrar o actualizar por id, así que este desajuste no puede ocurrir.


## T5 · El costo de la fuerza bruta

_Mi respuesta / análisis:_

1.000.000 vectores × 768 dimensiones ≈ 7,68 × 10⁸ multiplicaciones por consulta.
Con 100 consultas por segundo: 7,68 × 10⁸ × 100 = **7,68 × 10¹⁰ multiplicaciones por segundo**.

Ser exacto deja de ser viable porque ese costo crece linealmente con el número de vectores: cada consulta tiene que compararse contra todo el corpus, y con más datos o más usuarios la latencia y el hardware necesario se disparan.

En ANN se sacrifica un poco de exactitud (a veces no se devuelve el vecino más cercano real, recall < 100 %) y a cambio se gana muchísima velocidad, porque solo se visita una fracción pequeña de los vectores.


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_ [notebooks/02b_rag_app_nvidia_nim_colab.ipynb](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) (celdas ejecutadas con sus salidas).

_Mi respuesta / análisis:_

| Paso | Resultado en mi ejecución |
|---|---|
| Configuración NVIDIA NIM | ✅ Configuración lista, key leída desde los Secrets de Colab |
| Documentos | No subí `company_documents.json`, así que se usaron los **4 documentos de demostración** |
| Pipeline | ✅ 4 documentos → **4 chunks** indexados en Qdrant (en memoria) |
| Interfaz | ✅ Gradio publicado con enlace público temporal `*.gradio.live` |

- El pipeline completo corrió: embeddings (`nemotron-3-embed-1b`), búsqueda en Qdrant y respuesta del LLM (`nemotron-3-super-120b-a12b`).
- Como los documentos de demostración son muy cortos, el splitter no los partió: cada documento es un chunk. Con documentos más largos sí habría varios chunks por documento.
- Apareció la advertencia *"type is unknown and inference may fail"* para el modelo de embeddings, pero el índice se construyó sin errores.
- El `KeyboardInterrupt` al final del notebook es porque detuve manualmente la celda de Gradio (con `debug=True` la celda queda corriendo indefinidamente); no es un fallo de la app.


## A2 · Tu propio corpus de documentos

_Evidencia:_ …

_Mi respuesta / análisis:_


## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ …

_Mi respuesta / análisis:_


## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ [notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) (celdas ejecutadas con sus salidas).

_Mi respuesta / análisis:_

**Configuración de mi ejecución**

| Parámetro | Valor |
|---|---|
| Papers | *Attention Is All You Need* (40.174 caracteres) y *Retrieval-Augmented Generation* (69.121 caracteres) |
| Chunks | 55 (Attention) + 95 (RAG) = **150 chunks** en Qdrant |
| Embeddings | `nvidia/nemotron-3-embed-1b`, dimensión **2048** |
| Generador | `nvidia/nemotron-3.5-lightning-30b-a3b`, `max_completion_tokens=4000` |
| Retriever | `similarity_score_threshold`, `k=5`, `score_threshold=0.25` |

**Comparación con el ground truth**

| # | Pregunta | Respuesta del RAG | Ground truth | ¿Correcta? | ¿Contexto del paper correcto? |
|---|---|---|---|:---:|:---:|
| 1 | BLEU del Transformer (big) en WMT14 EN-DE | 28.4, citando *Attention Is All You Need* | 28.4 | Sí | Sí (Attention) |
| 2 | Cómo combina RAG memoria paramétrica y no paramétrica | Seq2seq preentrenado (BART) + índice denso de Wikipedia accedido con DPR, entrenado end-to-end | Igual | Sí | Sí (RAG) |
| 3 | Para qué sirve el factor √dk | Evita que los productos punto crezcan y saturen el softmax (gradientes muy pequeños) | Igual | Sí | Sí (Attention) |
| 4 | Variantes de RAG | RAG-Sequence (mismo documento para toda la secuencia) y RAG-Token (documento distinto por token) | Igual | Sí (no menciona que RAG-Token es más costoso) | Sí (RAG) |
| 5 | Cabezas y dimensiones del Transformer base | h = 8, dk = dv = 64 | h = 8, dk = dv = 64 | Sí | Sí (Attention) |
| Bonus | Generador de RAG y si es Transformer | BART, un seq2seq transformer | BART-large, basado en Transformer | Parcial (dice "BART", no "BART-large") | Ambos: 4 fragmentos de RAG + 1 de Attention |

**Resultado: 5/5 preguntas correctas y la bonus correcta en lo esencial (parcial solo por no decir "BART-large").**

**Problemas que encontré y cómo los resolví**

- El modelo de generación original del notebook devolvía **404 Not Found** para mi cuenta, y otros modelos probados (`meta/llama-3.1-70b-instruct`, `meta/llama-3.1-8b-instruct`, `qwen/qwen3-next-80b-a3b-instruct`…) devolvían **410 Gone** (fin de vida). Probé una lista de candidatos y el que respondió fue `nvidia/nemotron-3.5-lightning-30b-a3b`.
- Con `max_completion_tokens=800` casi todas las respuestas salían **vacías**: es un modelo de razonamiento y gastaba el presupuesto de tokens "pensando". Al subirlo a **4000** todas las respuestas aparecieron.

**Observaciones**

- Todas las respuestas citan el paper correcto y varias incluyen la frase textual del paper, lo que muestra que el modelo se apoya en el contexto recuperado y no en memoria propia.
- En la pregunta bonus, de los 5 fragmentos recuperados, 4 fueron del paper de RAG y 1 del de Attention: la recuperación sí trajo los dos papers, aunque la respuesta final se apoyó sobre todo en el de RAG.


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ …

_Mi respuesta / análisis:_


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ …

_Mi respuesta / análisis:_


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ …

_Mi respuesta / análisis:_


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ …

_Mi respuesta / análisis:_


## C2 · Un tercer paper: el RAG que explica HNSW

_Evidencia:_ …

_Mi respuesta / análisis:_


## C3 · App Gradio para los papers con filtro por paper

_Evidencia:_ …

_Mi respuesta / análisis:_


## C4 · Evaluación automática de la recuperación

_Evidencia:_ …

_Mi respuesta / análisis:_


## C5 · MMR frente a similitud pura

_Evidencia:_ …

_Mi respuesta / análisis:_


## C6 · Persistencia del índice en disco

_Evidencia:_ …

_Mi respuesta / análisis:_

