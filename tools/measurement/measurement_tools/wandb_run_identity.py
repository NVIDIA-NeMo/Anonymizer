# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B run naming and tag identity policy."""

from __future__ import annotations

from measurement_tools.wandb_metadata import WandbRunMetadata
from measurement_tools.wandb_settings import (
    ResolvedWandbConfig,
    generated_wandb_tag,
    is_safe_wandb_tag,
)

__all__ = ["default_run_name", "effective_wandb_tags"]


def default_run_name(suite_id: str, metadata: WandbRunMetadata | None) -> str:
    git = metadata.git if metadata is not None else None
    if git is None:
        return suite_id
    commit = git.commit
    branch = git.branch
    if isinstance(commit, str) and commit:
        suffix = commit[:7]
        if isinstance(branch, str) and branch:
            return f"{suite_id} {branch}@{suffix}"
        return f"{suite_id} @{suffix}"
    if isinstance(branch, str) and branch:
        return f"{suite_id} {branch}"
    return suite_id


def effective_wandb_tags(
    settings: ResolvedWandbConfig,
    *,
    suite_id: str,
    metadata: WandbRunMetadata | None,
) -> list[str]:
    tags = [tag for tag in settings.effective_wandb_tags if is_safe_wandb_tag(tag)]
    suite_tag = generated_wandb_tag("suite", suite_id)
    if suite_tag is not None:
        tags.append(suite_tag)
    git = metadata.git if metadata is not None else None
    if git is not None:
        branch = git.branch
        dirty = git.dirty
        if isinstance(branch, str) and branch:
            branch_tag = generated_wandb_tag("branch", branch)
            if branch_tag is not None:
                tags.append(branch_tag)
        if isinstance(dirty, bool):
            tags.append("dirty" if dirty else "clean")
    return tags
