import math

from foundry_local_sdk import Configuration, FoundryLocalManager

from database import get_all_documents


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve_documents(question, embedding_client, top_k=3):
    documents = get_all_documents()

    question_response = embedding_client.generate_embedding(question)
    question_embedding = question_response.data[0].embedding

    results = []

    for document in documents:
        score = cosine_similarity(
            question_embedding,
            document["embedding"],
        )

        results.append(
            {
                "source": document["source"],
                "content": document["content"],
                "score": score,
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)

    return results[:top_k]


def build_context(retrieved_documents):
    context_parts = []

    for document in retrieved_documents:
        context_parts.append(
            f"[Source: {document['source']}]\n"
            f"{document['content']}"
        )

    return "\n\n".join(context_parts)


def main():
    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(
        "qwen3-embedding-0.6b"
    )
    chat_model = manager.catalog.get_model(
        "phi-3.5-mini"
    )


    print("Embedding model is being prepared...")

    embedding_model.download(
    lambda progress: print(
        f"\rEmbedding model: %{progress:.1f}",
        end="",
        flush=True,
    )
)

    print("\nChat model is being prepared...")

    chat_model.download(
    lambda progress: print(
        f"\rChat model: %{progress:.1f}",
        end="",
        flush=True,
    )
)

    print("Models are loading...")

    embedding_model.load()
    chat_model.load()

    embedding_client = embedding_model.get_embedding_client()
    chat_client = chat_model.get_chat_client()

    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 250

    print("\nLocal RAG Assistant is ready.")
    print("Type 'quit' to close the application.\n")

    try:
        while True:
            question = input("Question: ").strip()

            if question.lower() == "quit":
                break

            if not question:
                print("Please enter a question.\n")
                continue

            retrieved_documents = retrieve_documents(
                question,
                embedding_client,
            )

            context = build_context(retrieved_documents)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a technical support assistant. "
                        "Answer the user's question using only the "
                        "provided context. Do not use outside knowledge. "
                        "If the answer is not available in the context, "
                        "say: 'I do not have enough information in the "
                        "knowledge base.' Keep the answer concise.\n\n"
                        f"Context:\n{context}"
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]

            response = chat_client.complete_chat(messages)
            answer = response.choices[0].message.content

            print(f"\nAnswer: {answer}")

            print("\nRetrieved sources:")
            for document in retrieved_documents:
                print(
                    f"- {document['source']} "
                    f"(score: {document['score']:.4f})"
                )

            print()

    finally:
        chat_model.unload()
        embedding_model.unload()
        print("\nModels unloaded. Application closed.")


if __name__ == "__main__":
    main()