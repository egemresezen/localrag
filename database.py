import json
import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "rag.db"


def create_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )


def clear_documents():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM document_chunks")


def save_document(source, content, embedding):
    embedding_json = json.dumps(embedding)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO document_chunks (source, content, embedding)
            VALUES (?, ?, ?)
            """,
            (source, content, embedding_json),
        )


def get_all_documents():
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            """
            SELECT id, source, content, embedding
            FROM document_chunks
            """
        ).fetchall()

    return [
        {
            "id": row[0],
            "source": row[1],
            "content": row[2],
            "embedding": json.loads(row[3]),
        }
        for row in rows
    ]

def replace_document(source, contents, embeddings):
    if len(contents) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings must be equal."
        )

    records = [
        (
            source,
            content,
            json.dumps(embedding),
        )
        for content, embedding in zip(contents, embeddings)
    ]

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            DELETE FROM document_chunks
            WHERE source = ?
            """,
            (source,),
        )

        connection.executemany(
            """
            INSERT INTO document_chunks (
                source,
                content,
                embedding
            )
            VALUES (?, ?, ?)
            """,
            records,
        )


def delete_document(source):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            DELETE FROM document_chunks
            WHERE source = ?
            """,
            (source,),
        )

        
if __name__ == "__main__":
    create_database()
    print(f"Database created: {DATABASE_PATH}")