"""Tests for SQLite tracker."""

import sqlite3
import threading

import pytest

from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    RequestLog,
)
from mmrouter.tracker.logger import Tracker


def _make_log(model="claude-haiku", cost=0.001, fallback=False):
    return RequestLog(
        prompt_hash=RequestLog.hash_prompt("test"),
        classification=ClassificationResult(
            complexity="simple", category="factual", confidence=0.9
        ),
        model_used=model,
        completion=CompletionResult(
            content="ok",
            model=model,
            tokens_in=10,
            tokens_out=5,
            cost=cost,
            latency_ms=150.0,
        ),
        fallback_used=fallback,
    )


class TestTracker:
    def test_log_and_stats(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(_make_log())
        stats = tracker.get_stats()

        assert stats["total_requests"] == 1
        assert stats["total_cost"] == 0.001
        assert stats["avg_latency_ms"] == 150.0
        assert stats["model_distribution"]["claude-haiku"]["count"] == 1
        tracker.close()

    def test_multiple_models(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(_make_log(model="claude-haiku", cost=0.001))
        tracker.log(_make_log(model="claude-haiku", cost=0.001))
        tracker.log(_make_log(model="claude-sonnet", cost=0.01))

        stats = tracker.get_stats()
        assert stats["total_requests"] == 3
        assert stats["model_distribution"]["claude-haiku"]["count"] == 2
        assert stats["model_distribution"]["claude-sonnet"]["count"] == 1
        tracker.close()

    def test_fallback_count(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(_make_log(fallback=False))
        tracker.log(_make_log(fallback=True))
        tracker.log(_make_log(fallback=True))

        stats = tracker.get_stats()
        assert stats["fallback_count"] == 2
        tracker.close()

    def test_empty_stats(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        stats = tracker.get_stats()

        assert stats["total_requests"] == 0
        assert stats["total_cost"] == 0
        assert stats["model_distribution"] == {}
        tracker.close()

    def test_wal_mode(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        cur = tracker._conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode == "wal"
        tracker.close()

    def test_db_created_automatically(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        tracker = Tracker(db_path)
        assert db_path.exists()
        tracker.close()


class TestUnrecordedStreams:
    """MMR-3: unrecorded streams are counted, by reason."""

    def test_get_stats_carries_unrecorded_streams(self, tmp_path):
        """Self-check 9 (tracker half): get_stats() carries the count and the
        by-reason map, even with none recorded yet."""
        tracker = Tracker(tmp_path / "test.db")
        stats = tracker.get_stats()

        assert stats["unrecorded_streams"] == 0
        assert stats["unrecorded_streams_by_reason"] == {}
        tracker.close()

    def test_record_unrecorded_stream_counts_by_reason(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.record_unrecorded_stream("claude-haiku-4-5-20251001", "no_usage")
        tracker.record_unrecorded_stream("claude-haiku-4-5-20251001", "no_usage")
        tracker.record_unrecorded_stream(None, "budget_rejected")

        stats = tracker.get_stats()
        assert stats["unrecorded_streams"] == 3
        assert stats["unrecorded_streams_by_reason"] == {
            "no_usage": 2,
            "budget_rejected": 1,
        }
        tracker.close()

    def test_record_unrecorded_stream_allows_null_model(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.record_unrecorded_stream(None, "budget_rejected")

        row = tracker.connection.execute(
            "SELECT model, reason FROM unrecorded_streams"
        ).fetchone()
        assert row[0] is None
        assert row[1] == "budget_rejected"
        tracker.close()

    def test_log_from_another_thread_does_not_raise(self, tmp_path):
        """Self-check 10: Tracker.log called from a thread other than the one
        that opened the connection does not raise. This is the scenario the
        streaming recorder hits on every normal (non-abort) completion."""
        import threading

        tracker = Tracker(tmp_path / "test.db")
        errors = []

        def write_from_thread():
            try:
                tracker.log(_make_log())
            except Exception as e:  # pragma: no cover - failure path only
                errors.append(e)

        thread = threading.Thread(target=write_from_thread)
        thread.start()
        thread.join()

        assert errors == []
        assert tracker.get_stats()["total_requests"] == 1
        tracker.close()

    def test_record_unrecorded_stream_from_another_thread_does_not_raise(self, tmp_path):
        import threading

        tracker = Tracker(tmp_path / "test.db")
        errors = []

        def write_from_thread():
            try:
                tracker.record_unrecorded_stream("claude-haiku-4-5-20251001", "aborted")
            except Exception as e:  # pragma: no cover - failure path only
                errors.append(e)

        thread = threading.Thread(target=write_from_thread)
        thread.start()
        thread.join()

        assert errors == []
        assert tracker.get_stats()["unrecorded_streams"] == 1
        tracker.close()


class TestUnrecordedStreamReasonConstraint:
    """MMR-3: `reason` is a closed set, enforced in the schema the way
    `feedback.rating` already is (CHECK (rating IN (-1, 1)) in the same file)."""

    def test_every_reason_the_code_emits_is_accepted(self, tmp_path):
        tracker = Tracker(tmp_path / "t.db")
        for reason in (
            "no_usage", "provider_error", "provider_unavailable",
            "aborted", "log_failed", "budget_rejected",
        ):
            tracker.record_unrecorded_stream("claude-haiku-4-5-20251001", reason)
        assert tracker.get_stats()["unrecorded_streams"] == 6
        tracker.close()

    def test_reason_outside_the_set_is_rejected(self, tmp_path):
        tracker = Tracker(tmp_path / "t.db")
        with pytest.raises(sqlite3.IntegrityError):
            tracker.record_unrecorded_stream("claude-haiku-4-5-20251001", "typo_reason")
        assert tracker.get_stats()["unrecorded_streams"] == 0
        tracker.close()


class TestConcurrentWritesUnderTheLock:
    """MMR-3: the connection is opened check_same_thread=False and guarded by one
    write lock. A single cross-thread write proves the flag; it does not prove
    two threads racing stay correct, and the lock is this ticket's riskiest part.

    Measured 2026-09-01 with the lock replaced by contextlib.nullcontext():
    83 of 480 writes landed and 7 of 8 threads raised
    `InterfaceError: bad parameter or other API misuse`. So sqlite3.threadsafety
    == 3 alone is NOT sufficient here -- this test discriminates.
    """

    def test_interleaved_writes_from_many_threads_lose_nothing(self, tmp_path):
        tracker = Tracker(tmp_path / "race.db")
        n_threads, per_thread = 8, 20
        barrier = threading.Barrier(n_threads)
        errors: list[str] = []

        def work(tid):
            try:
                barrier.wait()  # release together, maximise real overlap
                for i in range(per_thread):
                    if i % 2 == 0:
                        tracker.log(_make_log())
                    else:
                        tracker.record_unrecorded_stream("claude-haiku", "no_usage")
            except Exception as e:  # noqa: BLE001 -- the assertion is "none of these"
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = tracker.get_stats()
        assert errors == []
        assert stats["total_requests"] == n_threads * (per_thread // 2)
        assert stats["unrecorded_streams"] == n_threads * (per_thread - per_thread // 2)
        tracker.close()
