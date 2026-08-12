from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import re
import numpy as np

# Read the PDF file
reader = PdfReader("Code Travail Tunisie 2020.pdf")

text = ""
# Remove common watermark text that appears on every page (robust to spacing/case/accent)
watermark_re = re.compile(r"imprimerie\s+officielle(?:\s+de\s+la\s+r[eé]publique\s+tunisienne)?", re.I)
for page in reader.pages:
    page_text = page.extract_text() or ""
    # remove watermark occurrences from the page text
    page_text = watermark_re.sub(" ", page_text)
    text += page_text + " "

# Normalize whitespace (remove newlines and collapse multiple spaces)
text = re.sub(r"\s+", " ", text).strip()

#print(text[:1000])

# Embed the text using a pre-trained model

model = SentenceTransformer('all-MiniLM-L6-v2')

'''embeddings = model.encode(
    "Bonjour le monde je vais bien merci et toi"
)'''

#print(embeddings)
#print(embeddings.shape)

# def chunk_text(text, chunk_size=500):
#     chunks = []
#     for i in range(0, len(text), chunk_size):
#         chunks.append(text[i:i + chunk_size])
#     return chunks

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_sentences(text, chunk_size=500):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# test_text = "Bonjour le monde. Je vais bien, merci. Et toi? J'espère que tout va bien. C'est une belle journée pour apprendre le traitement du langage naturel."
# print(chunk_sentences(test_text, chunk_size=50))

chunks = chunk_sentences(text, chunk_size=500)
embeddings = model.encode(chunks)
#print(f"Number of chunks: {len(chunks)}")
#print (f"Shape of embeddings: {embeddings.shape}")

documents = []

for chunk, embedding in zip(chunks, embeddings):
    documents.append({
        "chunk": chunk,
        "embedding": embedding
    })

#print(documents[:3])

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0

def search(query, top_k=5):
    query_embedding = model.encode(query)

    results = []
    for document in documents:
        similarity = cosine_similarity(query_embedding, document["embedding"])
        results.append({"similarity": similarity, "document": document})

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:top_k]

def search_matmul(query, top_k=5):
    query_embedding = model.encode(query)
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = embeddings_norm @ query_embedding_norm
    top_indices = np.argsort(scores)[::-1][:top_k]

    # if large number of documents, consider using np.argpartition for efficiency
    # top_indices = np.argpartition(
    #     scores,
    #     -top_k
    # )[-top_k:]
    
    results = []
    for idx in top_indices:
        results.append({"similarity": scores[idx], "document": documents[idx]})
    return results

query = input("Enter your query: ")
top_results = search_matmul(query, top_k=5)
for result in top_results:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Document chunk: {result['document']['chunk']}\n")