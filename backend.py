import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai import RateLimitError

# ---------------- OpenAI Client ----------------
client = OpenAI()

# ---------------- Embedding Model ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def process_pdf(pdf_file):
    """
    Extract text from PDF and create chunks
    """
    # ✅ CRITICAL FIX: read bytes safely every time
    pdf_bytes = pdf_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    chunks = []
    chunk_id = 0

    for page in doc:
        text = page.get_text("text").replace("\n", " ")

        sentences = text.split(". ")
        buffer = ""

        for sent in sentences:
            buffer += sent + ". "

            if len(buffer) > 350:
                chunks.append({
                    "id": chunk_id,
                    "page": page.number + 1,
                    "text": buffer.strip()
                })
                chunk_id += 1
                buffer = ""

        if buffer.strip():
            chunks.append({
                "id": chunk_id,
                "page": page.number + 1,
                "text": buffer.strip()
            })
            chunk_id += 1

    doc.close()
    return chunks


def build_faiss(chunks):
    """
    Build FAISS index from chunks
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index


def ask_question(question, chunks, index, top_k=15):
    """
    Answer question using FAISS + OpenAI
    """
    try:
        q_embedding = embed_model.encode([question])
        _, indices = index.search(np.array(q_embedding), top_k)

        context = [
            f"(Page {chunks[i]['page']}) {chunks[i]['text']}"
            for i in indices[0]
        ]

        context_text = "\n\n".join(context)

        prompt = f"""
You are an AI tutor for school students.

Answer the question using ONLY the textbook content below.
The answer MAY be paraphrased but MUST be based on the text.

If the answer is not present, say:
"Sorry, I couldn't find the answer in the textbook."

Textbook Content:
{context_text}

Question: {question}

Answer in clear bullet points:
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content

    except RateLimitError:
        return "⏳ Rate limit reached. Please wait 20 seconds and try again."