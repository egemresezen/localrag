import math
from foundry_local_sdk import Configuration, FoundryLocalManager


documents = [
    "Foundry Local runs artificial intelligence models directly on the device.",
    "SQLite is a lightweight database that stores data in a single local file.",
    "RAG retrieves relevant documents before generating an answer.",
    "Embeddings represent the semantic meaning of text as numerical vectors.",
]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def main():
    config = Configuration(app_name="embedding_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")

    model.download(
        lambda progress: print(
            f"\rEmbedding modeli indiriliyor: %{progress:.1f}",
            end="",
            flush=True,
        )
    )

    print("\nEmbedding modeli yükleniyor...")
    model.load()

    embedding_client = model.get_embedding_client()

    document_response = embedding_client.generate_embeddings(documents)
    document_embeddings = [
        item.embedding for item in document_response.data
    ]

    question = "Which database is suitable for local storage?"

    question_response = embedding_client.generate_embedding(question)
    question_embedding = question_response.data[0].embedding

    results = []

    for index, document_embedding in enumerate(document_embeddings):
        score = cosine_similarity(
            question_embedding,
            document_embedding,
        )
        results.append((score, documents[index]))

    results.sort(key=lambda result: result[0], reverse=True)

    print(f"\nQuestion: {question}")
    print(f"Most relevant document: {results[0][1]}")
    print(f"Similarity score: {results[0][0]:.4f}")

    model.unload()


if __name__ == "__main__":
    main()