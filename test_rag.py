from lingollm.rag import ingest_grammar, retrieve_grammar, get_rag_context

GRAMMAR_PATH = "data/manchu/manchu_grammar.md"

ingest_grammar(GRAMMAR_PATH, language="manchu")

results = retrieve_grammar("tere bithe be hendume", "manchu", k=3)

for content, score in results:
    print("Similarity: ", score)
    print(content[:500])
    print("-" * 50)

context = get_rag_context("tere bithe be hendume", "manchu", k=3)

print("\nRAG Context:\n")
print(context[:1500])