import sqlite3
from pathlib import Path


class ConversationMemory:
    def __init__(self, path="data/conversations.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.connection.commit()

    def remember(self, question, answer):
        self.connection.execute(
            "INSERT INTO conversations (question, answer) VALUES (?, ?)",
            (question, answer)
        )
        self.connection.commit()

    def recent(self, limit=5):
        cursor = self.connection.execute(
            """
            SELECT question, answer
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        return list(reversed(cursor.fetchall()))

    def close(self):
        self.connection.close()
