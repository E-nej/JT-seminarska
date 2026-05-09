import psycopg
from sentence_transformers import SentenceTransformer

DB_URL = "postgresql://raguser:ragpass@localhost:5432/ragdb"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)

# razdeli slovnico na manjše prekrivajoče se dele
def chunk_text(text: str, size: int = 1200, overlap: int = 200):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap  #premik naprej z overlapom

    return chunks


def ingest_grammar(path: str, language: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)  # slovnico razdeli na chunks
    embeddings = model.encode(chunks, normalize_embeddings=True)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # izbris starih chunks, če ponovno nalagamo isto slovnico
            cur.execute(
                "DELETE FROM rag_chunks WHERE language = %s AND source = %s",
                (language, path)
            )

            # shranjevanje novih chunks in njihovih embeddingov v bazo
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO rag_chunks
                    (language, source, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (language, path, i, chunk, emb.tolist())
                )

        conn.commit()


def retrieve_grammar(query: str, language: str, k: int = 3):
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()  # vhodni stavek pretvori v embedding

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, 1 - (embedding <=> %s::vector) AS similarity
                FROM rag_chunks
                WHERE language = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, language, query_embedding, k)
            )

            return cur.fetchall()
        
# vrne besedilo chunkov, pripravljeni za vstavljanje v prompt, združeni z ločilom
def get_rag_context(query: str, language: str, k: int = 3) -> str:
    results = retrieve_grammar(query, language, k)
    return "\n---\n".join(content for content, similarity in results)