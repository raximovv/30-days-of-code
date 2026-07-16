import os
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def find_pdf():
    files = [f for f in os.listdir() if f.lower().endswith(".pdf")]
    if files:
        return files[0]
    path = input("Enter PDF path: ").strip()
    return path

def extract_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    doc = fitz.open(path)
    data = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            data.append((i + 1, text))
    return data

def chunk_text(pages):
    chunks = []
    for page_num, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk = " ".join(words[start:end])
            chunks.append({"text": chunk, "page": page_num})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_and_store(chunks):
    client = chromadb.Client()
    collection = client.get_or_create_collection("datasheets")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [str(i) for i in range(len(texts))]
    metadatas = [{"page": c["page"]} for c in chunks]
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return collection, model

def retrieve(query, collection, model, k=3):
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
    return results["documents"][0], results["metadatas"][0]

def generate_answer(query, docs, metas):
    context = ""
    for d, m in zip(docs, metas):
        context += f"(Page {m['page']}) {d}\n"
    prompt = f"Answer the question using the context and cite page numbers.\n\n{context}\nQuestion: {query}\nAnswer:"
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def main():
    path = find_pdf()
    print(f"Using: {path}")
    pages = extract_pdf(path)
    chunks = chunk_text(pages)
    collection, model = embed_and_store(chunks)
    while True:
        query = input("Ask: ")
        docs, metas = retrieve(query, collection, model)
        answer = generate_answer(query, docs, metas)
        print("\n" + answer + "\n")

if __name__ == "__main__":
    main()