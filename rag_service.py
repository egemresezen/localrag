import math

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from database import (
    get_all_documents,
    replace_document,
)
from document_processor import process_document


class RagService:
    def __init__(self):
        print("[1/5] Foundry Local is starting...")

        config = Configuration(
            app_name="localdoc_ai"
        )
        FoundryLocalManager.initialize(config)

        self.manager = FoundryLocalManager.instance
        catalog = self.manager.catalog

        print(
            "[2/5] Cached models are being checked..."
        )

        cached_models = catalog.get_cached_models()

        def get_cached_model(alias):
            model = catalog.get_model(alias)

            candidates = [
                cached_model
                for cached_model in cached_models
                if cached_model.alias.lower()
                == alias.lower()
            ]

            if candidates:
                cpu_variant = next(
                    (
                        candidate
                        for candidate in candidates
                        if "cpu" in candidate.id.lower()
                    ),
                    candidates[0],
                )

                model.select_variant(cpu_variant)

            else:
                print(
                    f"{alias} is not cached. Downloading..."
                )

                model.download(
                    lambda progress: print(
                        f"\r{alias}: %{progress:.1f}",
                        end="",
                        flush=True,
                    )
                )

            print(f"Selected model: {model.id}")
            return model

        self.embedding_model = get_cached_model(
            "qwen3-embedding-0.6b"
        )

        self.chat_model = get_cached_model(
            "phi-3.5-mini"
        )

        print(
            "[3/5] Embedding model is loading..."
        )

        if not self.embedding_model.is_loaded:
            self.embedding_model.load()

        print(
            "[4/5] Chat model is loading..."
        )

        if not self.chat_model.is_loaded:
            self.chat_model.load()

        self.embedding_client = (
            self.embedding_model.get_embedding_client()
        )

        self.chat_client = (
            self.chat_model.get_chat_client()
        )

        self.chat_client.settings.temperature = 0.0
        self.chat_client.settings.max_tokens = 300

        print("[5/5] RAG service is ready.")

    @staticmethod
    def cosine_similarity(vector_a, vector_b):
        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(
                value * value
                for value in vector_a
            )
        )

        magnitude_b = math.sqrt(
            sum(
                value * value
                for value in vector_b
            )
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return (
            dot_product
            / (magnitude_a * magnitude_b)
        )

    def retrieve(self, question, top_k=3):
        documents = get_all_documents()

        question_response = (
            self.embedding_client.generate_embedding(
                question
            )
        )

        question_embedding = (
            question_response.data[0].embedding
        )

        results = []

        for document in documents:
            score = self.cosine_similarity(
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

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:top_k]

    def answer(self, question, top_k=3):
        retrieved_documents = self.retrieve(
            question,
            top_k=top_k,
        )

        context = "\n\n".join(
            (
                f"[Source: {document['source']}]\n"
                f"{document['content']}"
            )
            for document in retrieved_documents
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document question-answering "
                    "assistant. Answer using only the provided "
                    "context. Do not use outside knowledge. "
                    "If the answer is not available, say: "
                    "'I do not have enough information in the "
                    "knowledge base.' Keep the answer concise."
                    "\n\n"
                    f"Context:\n{context}"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        response = self.chat_client.complete_chat(
            messages
        )

        answer_text = (
            response.choices[0].message.content
        )

        return answer_text, retrieved_documents

    def index_document(
        self,
        file_name,
        file_bytes,
        batch_size=8,
    ):
        chunks = process_document(
            file_name,
            file_bytes,
        )

        if not chunks:
            raise ValueError(
                "No readable content was found "
                "in the document."
            )

        embeddings = []
        total_batches = (
            len(chunks) + batch_size - 1
        ) // batch_size

        for batch_number, start_index in enumerate(
            range(0, len(chunks), batch_size),
            start=1,
        ):
            batch = chunks[
                start_index:start_index + batch_size
            ]

            print(
                "Generating embeddings: "
                f"batch {batch_number}/{total_batches}"
            )

            embedding_response = (
                self.embedding_client.generate_embeddings(
                    batch
                )
            )

            batch_embeddings = [
                item.embedding
                for item in embedding_response.data
            ]

            embeddings.extend(batch_embeddings)

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Some document chunks could not "
                "be embedded."
            )

        replace_document(
            source=file_name,
            contents=chunks,
            embeddings=embeddings,
        )

        return len(chunks)