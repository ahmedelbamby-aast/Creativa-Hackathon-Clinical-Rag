from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from backend import server
from src.embedding_quota import (
    EmbeddingQuotaController,
    EmbeddingQuotaExceeded,
    QuotaLimits,
    QuotaSnapshot,
    estimate_input_tokens,
    pacific_day_start,
    quota_block_reason,
)


class MemoryRepository:
    def __init__(self, snapshots: list[QuotaSnapshot] | None = None) -> None:
        self.snapshots = list(snapshots or [])
        self.events = []
        self.reservations = []

    def reserve(self, **values):
        self.reservations.append(values)
        return self.snapshots.pop(0)

    def record_event(self, **values): self.events.append(values)
    def quota_snapshot(self, limits, now):
        return QuotaSnapshot(1, 10, 2, limits.rpm, limits.tpm, limits.rpd, limits.safety_factor)
    def recent_runs(self, limit=20): return []
    def recent_events(self, limit=50): return self.events


def limits() -> QuotaLimits:
    return QuotaLimits(rpm=10, tpm=1000, rpd=100, safety_factor=0.70)


def test_pacific_day_boundary_tracks_daylight_saving_reset() -> None:
    winter = pacific_day_start(datetime(2026, 1, 15, 12, tzinfo=timezone.utc))
    summer = pacific_day_start(datetime(2026, 7, 15, 12, tzinfo=timezone.utc))
    assert winter.hour == 8
    assert summer.hour == 7


def test_effective_limits_and_alert_thresholds_are_distinct() -> None:
    assert limits().effective("rpm") == 7
    assert limits().effective("tpm") == 700
    assert limits().effective("rpd") == 95
    assert QuotaSnapshot(7, 0, 0, 10, 1000, 100, 0.7).alert == "warning"
    assert QuotaSnapshot(9, 0, 0, 10, 1000, 100, 0.7).alert == "critical"
    assert QuotaSnapshot(10, 0, 0, 10, 1000, 100, 0.7).alert == "hard_stop"


def test_rolling_rpm_and_tpm_reservations_stop_at_safety_budget() -> None:
    assert quota_block_reason({"rpm_used": 6, "tpm_used": 0, "rpd_used": 0}, limits(), 1) == ""
    assert quota_block_reason({"rpm_used": 7, "tpm_used": 0, "rpd_used": 0}, limits(), 1) == "rpm"
    assert quota_block_reason({"rpm_used": 0, "tpm_used": 690, "rpd_used": 0}, limits(), 11) == "tpm"


def test_interactive_quota_failure_is_immediate_and_recorded() -> None:
    blocked = QuotaSnapshot(7, 100, 3, 10, 1000, 100, 0.7, False, "rpm", 60)
    repository = MemoryRepository([blocked])
    controller = EmbeddingQuotaController(repository=repository, limits=limits())
    with pytest.raises(EmbeddingQuotaExceeded, match="rpm"):
        controller.acquire(["query"], "query", interactive=True)
    assert repository.events[0]["event_type"] == "quota_blocked"


def test_daily_limit_marks_ingestion_as_resumable() -> None:
    blocked = QuotaSnapshot(1, 10, 95, 10, 1000, 100, 0.7, False, "rpd", 0)
    controller = EmbeddingQuotaController(repository=MemoryRepository([blocked]), limits=limits())
    with pytest.raises(EmbeddingQuotaExceeded) as caught:
        controller.acquire(["document"], "document", interactive=False)
    assert caught.value.resumable is True


def test_input_token_estimate_is_conservative_for_arabic_and_english() -> None:
    assert estimate_input_tokens(["one two three", "مرحبا بالعالم"]) >= 7


def test_embedding_operations_endpoint_requires_constant_time_token(monkeypatch) -> None:
    monkeypatch.setattr(server.config, "operations_dashboard_token", "secret")
    monkeypatch.setattr(server.embedding_quota, "dashboard", lambda: {"configured": False})
    with pytest.raises(HTTPException) as missing:
        server.embedding_operations(authorization="")
    assert missing.value.status_code == 401
    assert server.embedding_operations(authorization="Bearer secret") == {"configured": False}
