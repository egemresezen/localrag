import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_text(file_name, file_bytes):
    extension = Path(file_name).suffix.lower()

    if extension == ".txt":
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

    if extension == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(
                    f"Page {page_number}\n{page_text.strip()}"
                )

        if not pages:
            raise ValueError(
                "No readable text was found in the PDF. "
                "Scanned PDFs require OCR support."
            )

        return "\n\n".join(pages)

    raise ValueError(
        "Unsupported file type. Only PDF and TXT are supported."
    )


def chunk_text(text, chunk_size=900, overlap=150):
    normalized_text = re.sub(r"\s+", " ", text).strip()

    if not normalized_text:
        raise ValueError("The document does not contain readable text.")

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    step_size = chunk_size - overlap

    chunks = [
        normalized_text[start:start + chunk_size]
        for start in range(
            0,
            len(normalized_text),
            step_size,
        )
    ]

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


def process_document(file_name, file_bytes):
    text = extract_text(file_name, file_bytes)
    return chunk_text(text)