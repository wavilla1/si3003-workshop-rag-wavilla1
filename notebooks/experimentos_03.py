# ===== EXPERIMENTOS B2, B3, B4, B5, C4, C5 (pegar al final del notebook 03) =====
import csv, os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

QS = [q1, q2, q3, q4, q5]
KEYS = [["28.4"], ["parametric"], ["softmax"], ["RAG-Sequence", "RAG-Token"], ["8", "64"]]
def correcta(ans, keys):
    return all(k.lower() in (ans or "").lower() for k in keys)

def build_index(chunk_size):
    overlap = int(chunk_size * 0.13)
    sp = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ".", " "])
    docs = []
    for text, title, src in [(attention_text, "Attention Is All You Need", "attention_paper.pdf"),
                             (rag_text, "Retrieval-Augmented Generation", "rag_paper.pdf")]:
        for i, c in enumerate(sp.split_text(text)):
            if len(c.strip()) >= 30:
                docs.append(Document(page_content=c, metadata={"title": title, "source": src, "chunk_index": i}))
    cl = QdrantClient(":memory:"); name = f"exp_{chunk_size}"
    cl.create_collection(name, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
    vs = QdrantVectorStore(client=cl, collection_name=name, embedding=embeddings)
    vs.add_documents(docs)
    return vs, len(docs)

def make_retr(vs, k=5, thr=0.25):
    return vs.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": thr})

def make_chain(retr, pr=None):
    return {"context": retr | format_docs, "question": RunnablePassthrough()} | (pr or prompt) | llm | StrOutputParser()

def ask(chain, q):
    try:
        return chain.invoke(q).strip()
    except Exception as e:
        return f"ERROR: {str(e)[:80]}"

def evaluar(vs, k=5, thr=0.25):
    retr = make_retr(vs, k, thr); ch = make_chain(retr)
    ok, nret = 0, []
    for q, keys in zip(QS, KEYS):
        nret.append(len(retr.invoke(q)))
        ok += correcta(ask(ch, q), keys)
    return ok, sum(nret) / len(nret)

# ---------- B2 ----------
print("=" * 20, "B2", "=" * 20)
INDEX = {}
for cs in (400, 900, 1500):
    INDEX[cs] = build_index(cs)
    ok, avg = evaluar(INDEX[cs][0])
    print(f"B2 chunk_size={cs:>4} chunks={INDEX[cs][1]:>3} k=5 thr=0.25 -> aciertos={ok}/5 docs_recuperados_prom={avg:.1f}")
for k in (3, 5, 8):
    ok, avg = evaluar(INDEX[900][0], k=k)
    print(f"B2 chunk_size= 900 k={k} thr=0.25 -> aciertos={ok}/5 docs_recuperados_prom={avg:.1f}")
for thr in (0.15, 0.25, 0.40):
    ok, avg = evaluar(INDEX[900][0], thr=thr)
    print(f"B2 chunk_size= 900 k=5 thr={thr} -> aciertos={ok}/5 docs_recuperados_prom={avg:.1f}")

# ---------- B3 ----------
print("=" * 20, "B3", "=" * 20)
TRAMPAS = [
    "What BLEU score did GPT-4 achieve on WMT 2014 English-to-German?",
    "How many parameters does the Llama 2 70B model have according to the RAG paper?",
    "What retrieval accuracy did RAG obtain on the MS MARCO 2023 leaderboard?",
]
base_retr = make_retr(INDEX[900][0]); base_chain = make_chain(base_retr)
for q in TRAMPAS:
    print("B3 Q:", q); print("B3 A:", ask(base_chain, q)[:400]); print()

# ---------- B4 ----------
print("=" * 20, "B4", "=" * 20)
ES = {
    "Q1": (q1, "¿Qué puntaje BLEU obtuvo el Transformer (modelo grande) en WMT 2014 inglés-alemán?"),
    "Q3": (q3, "¿Cuál es el propósito del factor de escala sqrt(dk) en el Scaled Dot-Product Attention?"),
    "Q4": (q4, "¿Cuáles son las dos variantes del modelo RAG propuestas en el paper y en qué se diferencian?"),
}
plain = INDEX[900][0]
for key, (en, es) in ES.items():
    ids_en = [(d.metadata["source"][:3], d.metadata["chunk_index"], round(s, 3)) for d, s in plain.similarity_search_with_score(en, k=5)]
    ids_es = [(d.metadata["source"][:3], d.metadata["chunk_index"], round(s, 3)) for d, s in plain.similarity_search_with_score(es, k=5)]
    comunes = len({x[:2] for x in ids_en} & {x[:2] for x in ids_es})
    print(f"B4 {key} EN chunks: {ids_en}")
    print(f"B4 {key} ES chunks: {ids_es}")
    print(f"B4 {key} chunks en común: {comunes}/5")
    print(f"B4 {key} respuesta ES: {ask(base_chain, es)[:300]}"); print()

# ---------- B5 ----------
print("=" * 20, "B5", "=" * 20)
B5 = [
    "How many identical layers does the Transformer encoder stack have?",
    "Which retriever does RAG use, and what document collection does it index?",
    "Is the generator used by RAG based on the Transformer architecture, and how many attention heads does the base Transformer use?",
]
for q in B5:
    docs = base_retr.invoke(q)
    print("B5 Q:", q)
    print("B5 A:", ask(base_chain, q)[:400])
    print("B5 fuentes:", [d.metadata["source"] for d in docs]); print()

# ---------- C4 ----------
print("=" * 20, "C4", "=" * 20)
os.makedirs("eval", exist_ok=True)
FILAS = [
    ("1", "How many attention heads does the base Transformer use?", "attention_paper.pdf", "parallel attention"),
    ("2", "What are the two RAG model variants?", "rag_paper.pdf", "RAG-Token"),
    ("3", "What BLEU score did the big Transformer achieve on English-to-German?", "attention_paper.pdf", "28.4"),
    ("4", "Why are dot products scaled in attention?", "attention_paper.pdf", "softmax"),
    ("5", "Which dense retriever does RAG use?", "rag_paper.pdf", "DPR"),
    ("6", "What seq2seq model is the RAG generator?", "rag_paper.pdf", "BART"),
    ("7", "How many layers does the Transformer encoder have?", "attention_paper.pdf", "identical layers"),
    ("8", "What optimizer was used to train the Transformer?", "attention_paper.pdf", "Adam"),
    ("9", "Which knowledge source is used as non-parametric memory in RAG?", "rag_paper.pdf", "Wikipedia"),
    ("10", "What positional encoding functions does the Transformer use?", "attention_paper.pdf", "sine"),
]
with open("eval/preguntas.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "pregunta", "fuente_esperada", "palabra_clave"]); w.writerows(FILAS)
filas = [dict(zip(["id", "pregunta", "fuente_esperada", "palabra_clave"], r)) for r in FILAS]

def hit_at_k(retriever, filas):
    hits = 0
    for r in filas:
        docs = retriever.invoke(r["pregunta"])
        hits += any(r["palabra_clave"].lower() in d.page_content.lower() and d.metadata["source"] == r["fuente_esperada"] for d in docs)
    return hits / len(filas)

for cs in (400, 900, 1500):
    vs = INDEX[cs][0]
    h3 = hit_at_k(vs.as_retriever(search_kwargs={"k": 3}), filas)
    h5 = hit_at_k(vs.as_retriever(search_kwargs={"k": 5}), filas)
    print(f"C4 chunk_size={cs:>4} hit@3={h3:.0%} hit@5={h5:.0%}")

# ---------- C5 ----------
print("=" * 20, "C5", "=" * 20)
sim_r = plain.as_retriever(search_kwargs={"k": 5})
mmr_r = plain.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5})
for q in [q2, q4, q_bonus]:
    s = sim_r.invoke(q); m = mmr_r.invoke(q)
    print("C5 Q:", q[:90])
    print("C5 similitud:", [(d.metadata["source"][:3], d.metadata["chunk_index"]) for d in s])
    print("C5 MMR      :", [(d.metadata["source"][:3], d.metadata["chunk_index"]) for d in m])
    print()
print("✅ Experimentos terminados")
