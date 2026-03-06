# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from anonymizer.logging import ProgressTracker


class TestProgressTracker:
    def test_logs_at_interval(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total=100, label="Replacement")
        with caplog.at_level(logging.INFO, logger="anonymizer"):
            for _ in range(10):
                tracker.record_success()
        assert "10/100" in caplog.text
        assert "10%" in caplog.text

    def test_no_log_before_interval(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total=100, label="Replacement")
        with caplog.at_level(logging.INFO, logger="anonymizer"):
            for _ in range(5):
                tracker.record_success()
        assert "progress" not in caplog.text

    def test_logs_final(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total=10, label="Replacement")
        with caplog.at_level(logging.INFO, logger="anonymizer"):
            for _ in range(10):
                tracker.record_success()
            tracker.log_final()
        assert "10/10" in caplog.text

    def test_tracks_failures(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total=20, label="Replacement")
        with caplog.at_level(logging.INFO, logger="anonymizer"):
            tracker.record_success()
            tracker.record_failure()
            for _ in range(18):
                tracker.record_success()
            tracker.log_final()
        assert "1 failed" in caplog.text

    def test_small_total_no_progress_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Under threshold (50), no interval progress logs emitted."""
        tracker = ProgressTracker(total=10, label="Replacement")
        with caplog.at_level(logging.INFO, logger="anonymizer"):
            for _ in range(10):
                tracker.record_success()
        # Only final log, no interval logs
        assert "progress" not in caplog.text
