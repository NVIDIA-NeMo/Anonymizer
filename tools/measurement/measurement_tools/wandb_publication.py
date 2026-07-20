# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated W&B publication envelopes and lifecycle results."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from measurement_tools.validation import StrictFrozenModel, VisibleIdentifier
from measurement_tools.wandb_metadata import WandbConfigPayload, WandbTag
from measurement_tools.wandb_metrics import WandbHistoryPayload, WandbSummaryPayload, WandbTablePayload
from measurement_tools.wandb_settings import WandbMode

__all__ = [
    "PUBLICATION_COMPLETE_KEY",
    "PUBLICATION_SEAL_DIGEST_KEY",
    "WandbInitPayload",
    "WandbPublicationState",
    "WandbPublishPayload",
    "WandbPublishResult",
]

PUBLICATION_COMPLETE_KEY = "publication/complete"
PUBLICATION_SEAL_DIGEST_KEY = "publication/completion_seal_sha256"


class WandbPublicationState(StrEnum):
    created = "created"
    resumed = "resumed"
    already_complete = "already_complete"


class WandbInitPayload(StrictFrozenModel):
    run_id: VisibleIdentifier
    resume: Literal["allow", "never"] = "never"
    project: StrictStr
    name: StrictStr
    mode: WandbMode
    directory: Path
    group: StrictStr
    job_type: StrictStr
    entity: StrictStr | None = None
    tags: tuple[WandbTag, ...] = ()


class WandbPublishPayload(StrictFrozenModel):
    init: WandbInitPayload
    config: WandbConfigPayload
    history: WandbHistoryPayload
    summary: WandbSummaryPayload
    tables: tuple[WandbTablePayload, ...] = ()

    @model_validator(mode="after")
    def validate_resume_contract(self) -> WandbPublishPayload:
        marker = self.summary.metrics.get(PUBLICATION_COMPLETE_KEY)
        digest = self.summary.metrics.get(PUBLICATION_SEAL_DIGEST_KEY)
        if self.init.resume == "allow":
            imported = self.config.imported
            if imported is None or marker is not True or digest != imported.completion_seal_sha256:
                raise ValueError("resumable W&B payload requires matching imported publication markers")
        elif marker is not None or digest is not None:
            raise ValueError("fresh W&B payload cannot contain resume publication markers")
        return self


class WandbPublishResult(StrictFrozenModel):
    published: StrictBool
    run_id: StrictStr | None = None
    entity: StrictStr | None = None
    project: StrictStr | None = None
    run_url: StrictStr | None = None
    publication_state: WandbPublicationState | None = None
    measurement_sha256: Annotated[StrictStr | None, Field(pattern=r"^[0-9a-f]{64}$")] = None
    record_count: StrictInt = 0
