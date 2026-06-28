"""
Tests for FeedbackDB — SQLite-backed feedback database.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feedback import FeedbackDB, FeedbackEntry, DBStats, _tokenize


@pytest.fixture
def db():
    """Create a temporary in-file database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    database = FeedbackDB.open(path)
    yield database
    os.unlink(path)


@pytest.fixture
def db_mem():
    """Create an in-memory database via temp file (SQLite needs a path)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    database = FeedbackDB.open(path)
    yield database
    os.unlink(path)


def _add_sample(
    db, prompt="a brave knight", rating=0, feedback=None, image_path="/tmp/test.png"
):
    entry_id = db.add(
        prompt=prompt,
        params={"seed": 42, "width": 512, "height": 512, "steps": 8},
        image_path=image_path,
    )
    if rating > 0:
        db.update_rating(entry_id, rating, feedback)
    return entry_id


class TestOpenAndInit:
    def test_open_creates_tables(self, db):
        stats = db.stats()
        assert stats.total == 0

    def test_open_creates_indexes(self, db):
        cursor = db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        names = [row[0] for row in cursor]
        assert "idx_prompt" in names
        assert "idx_rating" in names


class TestAdd:
    def test_add_returns_id(self, db):
        entry_id = db.add("knight", {"seed": 1}, "/tmp/k.png")
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    def test_add_increments_count(self, db):
        db.add("knight", {"seed": 1})
        db.add("archer", {"seed": 2})
        assert db.stats().total == 2

    def test_add_default_rating_is_zero(self, db):
        entry_id = db.add("knight", {"seed": 1})
        entries = db.get_all()
        assert entries[0].rating == 0

    def test_add_stores_params(self, db):
        params = {"seed": 42, "width": 512, "height": 512, "steps": 8}
        db.add("knight", params)
        entries = db.get_all()
        assert entries[0].params == params

    def test_add_stores_image_path(self, db):
        db.add("knight", {"seed": 1}, "/tmp/knight.png")
        entries = db.get_all()
        assert entries[0].image_path == "/tmp/knight.png"

    def test_add_without_image_path(self, db):
        db.add("knight", {"seed": 1}, None)
        entries = db.get_all()
        assert entries[0].image_path is None


class TestUpdateRating:
    def test_update_rating(self, db):
        entry_id = _add_sample(db)
        db.update_rating(entry_id, 5)
        entries = db.get_all()
        assert entries[0].rating == 5

    def test_update_rating_with_feedback(self, db):
        entry_id = _add_sample(db)
        db.update_rating(entry_id, 4, "great colors")
        entries = db.get_all()
        assert entries[0].rating == 4
        assert entries[0].feedback == "great colors"

    def test_update_rating_clamps_high(self, db):
        entry_id = _add_sample(db)
        db.update_rating(entry_id, 10)
        entries = db.get_all()
        assert entries[0].rating == 5

    def test_update_rating_clamps_negative(self, db):
        entry_id = _add_sample(db)
        db.update_rating(entry_id, -3)
        entries = db.get_all()
        assert entries[0].rating == 0

    def test_update_rating_to_zero(self, db):
        entry_id = _add_sample(db, rating=5)
        db.update_rating(entry_id, 0)
        entries = db.get_all()
        assert entries[0].rating == 0


class TestGetAll:
    def test_get_all_empty(self, db):
        assert db.get_all() == []

    def test_get_all_returns_entries(self, db):
        _add_sample(db, "knight")
        _add_sample(db, "archer")
        entries = db.get_all()
        assert len(entries) == 2

    def test_get_all_ordered_newest_first(self, db):
        id1 = _add_sample(db, "first")
        # Force different timestamp
        import time as _time

        _time.sleep(1.1)
        id2 = _add_sample(db, "second")
        entries = db.get_all()
        assert entries[0].prompt == "second"
        assert entries[1].prompt == "first"

    def test_get_all_returns_feedback_entry(self, db):
        _add_sample(db, "knight", rating=5, feedback="perfect")
        entry = db.get_all()[0]
        assert isinstance(entry, FeedbackEntry)
        assert entry.prompt == "knight"
        assert entry.rating == 5
        assert entry.feedback == "perfect"


class TestGetUnrated:
    def test_get_unrated_empty(self, db):
        assert db.get_unrated() == []

    def test_get_unrated_only_unrated(self, db):
        id1 = _add_sample(db, "rated", rating=5)
        id2 = _add_sample(db, "unrated")
        unrated = db.get_unrated()
        assert len(unrated) == 1
        assert unrated[0].prompt == "unrated"

    def test_get_unrated_all_unrated(self, db):
        _add_sample(db, "sprite1")
        _add_sample(db, "sprite2")
        assert len(db.get_unrated()) == 2


class TestTopRated:
    def test_top_rated_empty(self, db):
        assert db.top_rated(10, 1) == []

    def test_top_rated_filters_min_rating(self, db):
        _add_sample(db, "low", rating=1)
        _add_sample(db, "high", rating=5)
        top = db.top_rated(10, 4)
        assert len(top) == 1
        assert top[0].prompt == "high"

    def test_top_rated_orders_by_rating(self, db):
        _add_sample(db, "three", rating=3)
        _add_sample(db, "five", rating=5)
        _add_sample(db, "four", rating=4)
        top = db.top_rated(3, 1)
        assert top[0].rating == 5
        assert top[1].rating == 4
        assert top[2].rating == 3

    def test_top_rated_respects_limit(self, db):
        for i in range(10):
            _add_sample(db, f"sprite_{i}", rating=5)
        top = db.top_rated(3, 1)
        assert len(top) == 3


class TestSearchSimilar:
    def test_search_similar_empty_db(self, db):
        assert db.search_similar("knight", 5) == []

    def test_search_similar_exact_match(self, db):
        _add_sample(db, "brave knight", rating=5)
        _add_sample(db, "fire dragon", rating=4)
        results = db.search_similar("knight", 5)
        assert len(results) == 1
        assert "knight" in results[0].prompt

    def test_search_similar_multiple_keywords(self, db):
        _add_sample(db, "brave knight", rating=5)
        _add_sample(db, "brave warrior", rating=4)
        _add_sample(db, "fire dragon", rating=3)
        results = db.search_similar("brave knight", 5)
        assert len(results) == 2
        assert "knight" in results[0].prompt
        assert "warrior" in results[1].prompt

    def test_search_similar_no_match(self, db):
        _add_sample(db, "fire dragon", rating=5)
        results = db.search_similar("knight", 5)
        assert results == []

    def test_search_similar_empty_query_returns_top(self, db):
        _add_sample(db, "knight", rating=5)
        _add_sample(db, "dragon", rating=3)
        results = db.search_similar("", 5)
        assert len(results) == 2

    def test_search_similar_respects_limit(self, db):
        for i in range(10):
            _add_sample(db, f"knight variant {i}", rating=5)
        results = db.search_similar("knight", 3)
        assert len(results) == 3


class TestStats:
    def test_stats_empty(self, db):
        stats = db.stats()
        assert stats.total == 0
        assert stats.rated == 0
        assert stats.unrated == 0
        assert stats.avg_rating == 0.0

    def test_stats_with_entries(self, db):
        _add_sample(db, "s1", rating=4)
        _add_sample(db, "s2", rating=2)
        _add_sample(db, "s3")
        stats = db.stats()
        assert stats.total == 3
        assert stats.rated == 2
        assert stats.unrated == 1
        assert abs(stats.avg_rating - 3.0) < 0.1

    def test_stats_all_unrated(self, db):
        _add_sample(db, "s1")
        _add_sample(db, "s2")
        stats = db.stats()
        assert stats.rated == 0
        assert stats.unrated == 2
        assert stats.avg_rating == 0.0


class TestDelete:
    def test_delete_entry(self, db):
        entry_id = _add_sample(db, "knight")
        assert db.stats().total == 1
        db.delete(entry_id)
        assert db.stats().total == 0

    def test_delete_nonexistent_id(self, db):
        db.delete("nonexistent-id")
        assert db.stats().total == 0

    def test_delete_specific_entry(self, db):
        id1 = _add_sample(db, "knight")
        id2 = _add_sample(db, "archer")
        db.delete(id1)
        entries = db.get_all()
        assert len(entries) == 1
        assert entries[0].prompt == "archer"


class TestExportJsonl:
    def test_export_jsonl(self, db, tmp_path):
        _add_sample(db, "knight", rating=5, feedback="great")
        _add_sample(db, "archer", rating=4)
        _add_sample(db, "goblin", rating=1)

        path = str(tmp_path / "export.jsonl")
        count = db.export_jsonl(path, min_rating=4)

        assert count == 2
        assert os.path.exists(path)

        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data = json.loads(lines[0])
        assert "instruction" in data
        assert "response" in data
        assert "rating" in data

    def test_export_jsonl_empty(self, db, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        count = db.export_jsonl(path, min_rating=4)
        assert count == 0

    def test_export_jsonl_min_rating_filter(self, db, tmp_path):
        _add_sample(db, "high", rating=5)
        _add_sample(db, "mid", rating=3)
        _add_sample(db, "low", rating=1)

        path = str(tmp_path / "filter.jsonl")
        count = db.export_jsonl(path, min_rating=3)
        assert count == 2


class TestTokenize:
    def test_simple_words(self):
        assert _tokenize("knight armor") == ["knight", "armor"]

    def test_underscores(self):
        assert _tokenize("missile_launch") == ["missile", "launch"]

    def test_hyphens(self):
        assert _tokenize("fire-ball") == ["fire", "ball"]

    def test_single_char_filtered(self):
        assert _tokenize("a b c") == []

    def test_mixed_case(self):
        assert _tokenize("Brave Knight") == ["brave", "knight"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_separators(self):
        assert _tokenize("_ - _") == []
