import streamlit as st

from database import delete_document, get_all_documents
from rag_service import RagService


st.set_page_config(
    page_title="LocalDoc AI",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_rag_service():
    return RagService()


if "messages" not in st.session_state:
    st.session_state.messages = []


if "notification" in st.session_state:
    st.success(st.session_state.pop("notification"))


documents = get_all_documents()

source_names = sorted(
    {document["source"] for document in documents}
)


st.title("📚 LocalDoc AI")
st.caption(
    "Upload PDF or TXT documents and ask questions "
    "using a fully local RAG application."
)


with st.sidebar:
    st.header("System")

    st.success("Local mode")
    st.metric("Documents", len(source_names))
    st.metric("Document chunks", len(documents))

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button(
        "Clear chat history",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


chat_tab, documents_tab = st.tabs(
    ["💬 Chat", "📄 Knowledge Base"]
)


with chat_tab:
    if not documents:
        st.warning(
            "The knowledge base is empty. "
            "Upload a document from the Knowledge Base tab."
        )

    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                if message.get("sources"):
                    with st.expander("Retrieved sources"):
                        for source in message["sources"]:
                            st.caption(
                                f"{source['source']} · "
                                f"Similarity: "
                                f"{source['score']:.4f}"
                            )
                            st.write(source["content"])

        question = st.chat_input(
            "Ask a question about the documents..."
        )

        if question:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching documents and generating "
                    "an answer..."
                ):
                    try:
                        rag_service = get_rag_service()

                        answer, sources = rag_service.answer(
                            question,
                            top_k=top_k,
                        )

                        st.write(answer)

                        with st.expander("Retrieved sources"):
                            for source in sources:
                                st.caption(
                                    f"{source['source']} · "
                                    f"Similarity: "
                                    f"{source['score']:.4f}"
                                )
                                st.write(source["content"])

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )

                    except Exception as error:
                        st.error(
                            f"An error occurred: {error}"
                        )


with documents_tab:
    st.subheader("Add Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help=(
            "Text-based PDFs are supported. "
            "Scanned PDFs require OCR."
        ),
    )

    add_button = st.button(
        "Add to Knowledge Base",
        type="primary",
        disabled=not uploaded_files,
    )

    if add_button:
        try:
            rag_service = get_rag_service()
            indexed_files = []

            with st.spinner(
                "Reading documents and generating embeddings..."
            ):
                for uploaded_file in uploaded_files:
                    chunk_count = (
                        rag_service.index_document(
                            file_name=uploaded_file.name,
                            file_bytes=uploaded_file.getvalue(),
                        )
                    )

                    indexed_files.append(
                        f"{uploaded_file.name} "
                        f"({chunk_count} chunks)"
                    )

            st.session_state.messages = []
            st.session_state["notification"] = (
                "Documents added successfully: "
                + ", ".join(indexed_files)
            )

            st.rerun()

        except Exception as error:
            st.error(
                f"Document could not be added: {error}"
            )

    st.divider()
    st.subheader("Indexed Documents")

    if not documents:
        st.info("No documents have been indexed.")

    else:
        for source_name in source_names:
            source_chunks = [
                document
                for document in documents
                if document["source"] == source_name
            ]

            with st.container(border=True):
                title_column, button_column = st.columns(
                    [5, 1]
                )

                with title_column:
                    st.markdown(
                        f"### {source_name}"
                    )
                    st.caption(
                        f"{len(source_chunks)} indexed chunks"
                    )

                with button_column:
                    delete_clicked = st.button(
                        "Delete",
                        key=f"delete_{source_name}",
                        use_container_width=True,
                    )

                if delete_clicked:
                    delete_document(source_name)
                    st.session_state.messages = []
                    st.session_state["notification"] = (
                        f"{source_name} was deleted."
                    )
                    st.rerun()

                with st.expander("View document chunks"):
                    for index, chunk in enumerate(
                        source_chunks,
                        start=1,
                    ):
                        st.markdown(
                            f"**Chunk {index}**"
                        )
                        st.write(chunk["content"])