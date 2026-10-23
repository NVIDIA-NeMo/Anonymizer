# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A runner on an adapter without ``run_workflow`` raises that adapter's own AttributeError."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow


def _any(value: object) -> Any:
    return cast(Any, value)


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_rewrite_runner_on_adapter_without_run_workflow_raises_its_attribute_error(
    tmp_path: Any, preview: bool
) -> None:
    from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
    from tests.interface.test_rewrite_caller_thread import _close, _designer, _model_configs

    class Duck:
        pass

    designer, clients = _designer(tmp_path)
    anonymizer = Anonymizer(
        data_designer=designer, model_configs=_model_configs(), rewrite_runner=RewriteWorkflow(_any(Duck()))
    )
    path = tmp_path / "input.csv"
    pd.DataFrame({"biography": ["Alice travels.", "Alice stays."]}).to_csv(path, index=False)
    config = AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False)
    data = AnonymizerInput(source=str(path), text_column="biography")
    try:
        with pytest.raises(AttributeError, match="^'Duck' object has no attribute 'run_workflow'$"):
            if preview:
                anonymizer.preview(config=config, data=data, num_records=2)
            else:
                anonymizer.run(config=config, data=data)
    finally:
        _close(clients)
