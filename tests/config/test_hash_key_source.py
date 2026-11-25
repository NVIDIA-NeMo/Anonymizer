# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A CLI built from Hash reads the key from ANONYMIZER_HASH_KEY and offers no --key flag."""

from __future__ import annotations

from typing import Annotated

import cyclopts
import pytest

from anonymizer.config.replace_strategies import Hash


def _app() -> cyclopts.App:
    app = cyclopts.App(result_action="return_value", exit_on_error=False, print_error=False)

    @app.default
    def build(*, hash_: Annotated[Hash | None, cyclopts.Parameter(name="*")] = None) -> Hash | None:
        return hash_

    return app


def test_the_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "s3cr3t")
    built = _app()([])
    assert isinstance(built, Hash)
    assert built.key is not None and built.key.get_secret_value() == "s3cr3t"


def test_the_key_has_no_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "s3cr3t")
    with pytest.raises(cyclopts.CycloptsError, match="--key"):
        _app()(["--key", "on-the-command-line"])
