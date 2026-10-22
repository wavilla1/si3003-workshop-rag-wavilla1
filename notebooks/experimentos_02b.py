# ===== EXPERIMENTOS A1, A2, A3, A4 (notebook 02b) =====
import json, requests
from pathlib import Path

GENERATION_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"   # el modelo original da 404 en mi cuenta
RAW = "https://raw.githubusercontent.com/wavilla1/si3003-workshop-rag-wavilla1/main/"

PROMPT_ORIGINAL = (
    "Eres un asistente empresarial. Responde únicamente con información "
    "del contexto proporcionado. Si el contexto no contiene la respuesta, "
    "indica que no tienes esa información. Responde en español y de forma concisa."
)
PROMPT_NUEVO = (
    "Eres un asistente empresarial. Responde únicamente con información del contexto proporcionado. "
    "Reglas: (a) cita el título del documento usado entre corchetes, por ejemplo [Tiempos de entrega]; "
    "(b) si el contexto no alcanza para responder, responde exactamente: "
    "'No tengo esa información en los documentos disponibles.'; "
    "(c) responde en español en máximo 3 oraciones."
)

def build_pipeline2(documents, system_prompt=PROMPT_ORIGINAL, chunk_size=400, k=3, threshold=0.3):
    embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")
    llm = ChatNVIDIA(model=GENERATION_MODEL, temperature=0.2, top_p=0.95, max_completion_tokens=4000)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 10, separators=["\n\n", "\n", ".", " "])
    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            chunks.append(Document(page_content=text, metadata={"source_id": document["id"], "title": document["title"], "chunk_index": index, **document.get("metadata", {})}))
    dim = len(embeddings.embed_query("prueba"))
    client = QdrantClient(":memory:")
    client.create_collection(collection_name=COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    vectorstore.add_documents(chunks)
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": threshold})
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}")])
    fmt = lambda items: "\n\n---\n\n".join(f"[{i.metadata.get('title')}]\n{i.page_content}" for i in items)
    answer = ({"context": lambda x: fmt(x["documents"]), "question": lambda x: x["question"]} | prompt | llm | StrOutputParser())
    pipeline = RunnableParallel(question=RunnablePassthrough(), documents=retriever).assign(answer=answer)
    return pipeline, chunks, vectorstore

def preguntar(pipeline, q, tag):
    try:
        r = pipeline.invoke(q)
        fuentes = sorted({d.metadata["title"] for d in r["documents"]})
        print(f"{tag} Q: {q}\n{tag} A: {r['answer'].strip()[:350]}\n{tag} Fuentes: {fuentes}\n")
    except Exception as e:
        print(f"{tag} Q: {q}\n{tag} ERROR: {str(e)[:100]}\n")

# ---------- A1 ----------
print("=" * 20, "A1 (documentos de demostración)", "=" * 20)
p_demo, ch_demo, vs_demo = build_pipeline2(DEMO_DOCUMENTS)
for q in ["¿Cuál es el precio del plan Enterprise de CloudSync Pro?",
          "¿Cuántos días pueden trabajar remotamente los empleados?",
          "¿Qué pasa con los reembolsos mayores a $100?",
          "¿Qué ocurre durante la primera semana de onboarding?",
          "¿Cuál es el horario del gimnasio de la empresa?"]:
    preguntar(p_demo, q, "A1")

# ---------- A2 ----------
print("=" * 20, "A2 (mi corpus)", "=" * 20)
Path("data").mkdir(exist_ok=True)
Path("data/company_documents.json").write_text(requests.get(RAW + "data/company_documents.json").text, encoding="utf-8")
mi_corpus = load_documents(Path("data/company_documents.json"))
print("Longitud de los documentos (caracteres):", [len(d["content"]) for d in mi_corpus])
p_mio, ch_mio, vs_mio = build_pipeline2(mi_corpus)
print(f"{len(mi_corpus)} documentos -> {len(ch_mio)} chunks con chunk_size=400\n")
A2_PREGUNTAS = [
    "¿Cuánto cuesta imprimir en PETG?",
    "¿En cuánto tiempo entregan un pedido de 150 gramos?",
    "¿Qué colores de PLA tienen en inventario?",
    "¿Cada cuánto se reemplaza la boquilla de las impresoras?",
    "¿Qué formatos de archivo aceptan para cotizar?",
    "¿Hacen envíos internacionales?",
    "¿Cuál es el salario de los empleados del taller?",
]
for q in A2_PREGUNTAS:
    preguntar(p_mio, q, "A2")

# ---------- A3 ----------
print("=" * 20, "A3 (prompt antes/después)", "=" * 20)
p_nuevo, _, _ = build_pipeline2(mi_corpus, system_prompt=PROMPT_NUEVO)
for q in ["¿Cuánto cuesta imprimir en PETG?",              # fácil
          "¿Cuánto me cobran por una pieza grande y urgente?",  # ambigua
          "¿Hacen envíos internacionales?"]:                # fuera de contexto
    preguntar(p_mio, q, "A3-ANTES")
    preguntar(p_nuevo, q, "A3-DESPUÉS")

# ---------- A4 ----------
print("=" * 20, "A4 (scores de similitud)", "=" * 20)
for q in ["¿Cuánto cuesta imprimir en PETG?",
          "¿Qué impresoras tiene el taller?",
          "¿Qué métodos de pago aceptan?",
          "¿Hacen envíos internacionales?",
          "¿Cuál es la capital de Francia?"]:
    res = vs_mio.similarity_search_with_score(q, k=3)
    print("A4 Q:", q)
    for d, s in res:
        print(f"A4    {s:.3f}  {d.metadata['title']}")
    print()

# ---------- App Gradio con mi corpus, prompt nuevo y scores (para las capturas) ----------
def chat_scores(question, history):
    history = history or []
    question = (question or "").strip()
    if not question:
        return "", history, "Escribe una pregunta."
    r = p_nuevo.invoke(question)
    scored = vs_mio.similarity_search_with_score(question, k=3)
    fuentes = "**Fuentes (score de similitud):**\n" + "\n".join(f"- {d.metadata['title']} — `{s:.3f}`" for d, s in scored)
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": r["answer"].strip()}]
    return "", history, fuentes

gr.close_all()
with gr.Blocks(title="RAG Taller de impresión 3D") as demo2:
    gr.Markdown("# RAG · Taller de impresión 3D\nMi corpus de 8 documentos · prompt mejorado · scores de similitud")
    chatbot2 = gr.Chatbot(height=380, label="Chat")
    q_box = gr.Textbox(label="Pregunta", placeholder="Ej: ¿Cuánto cuesta imprimir en PETG?")
    fuentes_box = gr.Markdown()
    q_box.submit(chat_scores, inputs=[q_box, chatbot2], outputs=[q_box, chatbot2, fuentes_box])
demo2.queue().launch(share=True, debug=False)
print("✅ Experimentos 02b terminados. Abre el enlace público de arriba para las capturas.")
