from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from database import clear_documents, create_database, save_document


DOCUMENTS_PATH = Path(__file__).resolve().parent / "documents"


def read_document_chunks():
    chunks = []

    for file_path in sorted(DOCUMENTS_PATH.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        for paragraph in paragraphs:
            chunks.append(
                {
                    "source": file_path.name,
                    "content": paragraph,
                }
            )

    return chunks


def main():
    create_database()
    clear_documents()

    chunks = read_document_chunks()

    if not chunks:
        print("No documents were found.")
        return

    config = Configuration(app_name="rag_ingestion")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")

    model.download(
        lambda progress: print(
            f"\rEmbedding model: %{progress:.1f}",
            end="",
            flush=True,
        )
    )

    print("\nEmbedding model is loading...")
    model.load()

    try:
        embedding_client = model.get_embedding_client()

        contents = [chunk["content"] for chunk in chunks]
        response = embedding_client.generate_embeddings(contents)

        for chunk, embedding_result in zip(chunks, response.data):
            save_document(
                source=chunk["source"],
                content=chunk["content"],
                embedding=embedding_result.embedding,
            )

        print(f"{len(chunks)} document chunks were saved to SQLite.")
    finally:
        model.unload()


if __name__ == "__main__":
    main()