"""
Feedback database — SQLite-backed storage for sprite ratings.

Stores generated sprites with their prompts, params, PNG paths, and user ratings (1-5 stars).
Provides similarity search by prompt keywords for few-shot reference examples.
"""

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeedbackEntry:
    id: str
    prompt: str
    params: dict
    rating: int
    feedback: Optional[str]
    image_path: Optional[str]
    created_at: str


@dataclass
class DBStats:
    total: int = 0
    rated: int = 0
    unrated: int = 0
    avg_rating: float = 0.0


class FeedbackDB:
    """SQLite-backed feedback database for generated sprites."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @classmethod
    def open(cls, path: str) -> "FeedbackDB":
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                params_json TEXT NOT NULL,
                rating INTEGER DEFAULT 0,
                feedback TEXT,
                image_path TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt ON feedback(prompt)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rating ON feedback(rating)")
        conn.commit()
        return cls(conn)

    def add(
        self,
        prompt: str,
        params: dict,
        image_path: Optional[str] = None,
    ) -> str:
        entry_id = str(uuid.uuid4())
        params_json = json.dumps(params)
        now = str(int(time.time()))

        self.conn.execute(
            "INSERT INTO feedback (id, prompt, params_json, rating, feedback, image_path, created_at) "
            "VALUES (?, ?, ?, 0, NULL, ?, ?)",
            (entry_id, prompt, params_json, image_path, now),
        )
        self.conn.commit()
        return entry_id

    def update_rating(
        self,
        entry_id: str,
        rating: int,
        feedback: Optional[str] = None,
    ) -> None:
        rating = max(0, min(5, rating))
        self.conn.execute(
            "UPDATE feedback SET rating = ?, feedback = ? WHERE id = ?",
            (rating, feedback, entry_id),
        )
        self.conn.commit()

    def get_all(self) -> list[FeedbackEntry]:
        cursor = self.conn.execute(
            "SELECT id, prompt, params_json, rating, feedback, image_path, created_at "
            "FROM feedback ORDER BY created_at DESC"
        )
        return [self._row_to_entry(row) for row in cursor]

    def get_unrated(self) -> list[FeedbackEntry]:
        cursor = self.conn.execute(
            "SELECT id, prompt, params_json, rating, feedback, image_path, created_at "
            "FROM feedback WHERE rating = 0 ORDER BY created_at DESC"
        )
        return [self._row_to_entry(row) for row in cursor]

    def top_rated(self, limit: int = 10, min_rating: int = 1) -> list[FeedbackEntry]:
        cursor = self.conn.execute(
            "SELECT id, prompt, params_json, rating, feedback, image_path, created_at "
            "FROM feedback WHERE rating >= ? ORDER BY rating DESC, created_at DESC LIMIT ?",
            (min_rating, limit),
        )
        return [self._row_to_entry(row) for row in cursor]

    def search_similar(self, query: str, limit: int = 5) -> list[FeedbackEntry]:
        query_keywords = _tokenize(query)

        if not query_keywords:
            return self.top_rated(limit, 1)

        all_entries = self.get_all()

        scored = []
        for entry in all_entries:
            entry_keywords = _tokenize(entry.prompt)
            match_count = sum(
                1 for qk in query_keywords if any(ek == qk for ek in entry_keywords)
            )
            if match_count > 0:
                scored.append((entry, match_count, entry.rating))

        scored.sort(key=lambda x: (-x[1], -x[2]))
        return [e for e, _, _ in scored[:limit]]

    def stats(self) -> DBStats:
        total = self.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        rated = self.conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating > 0"
        ).fetchone()[0]
        avg = (
            self.conn.execute(
                "SELECT AVG(rating) FROM feedback WHERE rating > 0"
            ).fetchone()[0]
            or 0.0
        )

        return DBStats(
            total=total,
            rated=rated,
            unrated=total - rated,
            avg_rating=round(avg, 1),
        )

    def delete(self, entry_id: str) -> None:
        self.conn.execute("DELETE FROM feedback WHERE id = ?", (entry_id,))
        self.conn.commit()

    def export_jsonl(self, path: str, min_rating: int = 4) -> int:
        entries = self.top_rated(10000, min_rating)
        lines = []

        for entry in entries:
            line = json.dumps(
                {
                    "instruction": f"Generate a pixel-art sprite for: {entry.prompt}",
                    "response": entry.params,
                    "rating": entry.rating,
                }
            )
            lines.append(line)

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return len(entries)

    def _row_to_entry(self, row: sqlite3.Row) -> FeedbackEntry:
        return FeedbackEntry(
            id=row[0],
            prompt=row[1],
            params=json.loads(row[2]),
            rating=max(0, min(5, row[3])),
            feedback=row[4],
            image_path=row[5],
            created_at=row[6],
        )


def _tokenize(s: str) -> list[str]:
    """Tokenize a prompt into lowercase keywords."""
    return [
        w.lower() for w in s.replace("_", " ").replace("-", " ").split() if len(w) > 1
    ]
