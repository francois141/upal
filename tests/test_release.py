"""Focused tests for release-only safety checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.push_to_hub import verify_repo_visibility


class FakeHubApi:
    def __init__(self, private: bool):
        self.private = private

    def repo_info(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        assert repo_id == "owner/model"
        assert repo_type == "model"
        return SimpleNamespace(private=self.private)


@pytest.mark.parametrize("private", [False, True])
def test_repo_visibility_accepts_expected_state(private: bool) -> None:
    verify_repo_visibility(FakeHubApi(private), "owner/model", private=private)


@pytest.mark.parametrize("expected_private", [False, True])
def test_repo_visibility_rejects_unexpected_state(expected_private: bool) -> None:
    with pytest.raises(RuntimeError, match="change the repository visibility"):
        verify_repo_visibility(
            FakeHubApi(not expected_private),
            "owner/model",
            private=expected_private,
        )
