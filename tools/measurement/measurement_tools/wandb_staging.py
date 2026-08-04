# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secure filesystem staging for W&B artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path

__all__ = ["open_directory_no_follow", "prepare_wandb_staging_dir", "validate_directory_metadata"]


def prepare_wandb_staging_dir(output_dir: Path) -> Path:
    output_descriptor = open_directory_no_follow(output_dir)
    staging_dir = output_dir / ".wandb-private"
    try:
        try:
            os.mkdir(".wandb-private", mode=0o700, dir_fd=output_descriptor)
        except FileExistsError:
            pass
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        staging_descriptor = os.open(
            ".wandb-private",
            flags | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_descriptor,
        )
        try:
            metadata = os.fstat(staging_descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise ValueError("W&B staging directory must be an owned directory")
            os.fchmod(staging_descriptor, 0o700)
        finally:
            os.close(staging_descriptor)
    except OSError as exc:
        raise ValueError("W&B staging directory cannot contain symlinks or special files") from exc
    finally:
        os.close(output_descriptor)
    return staging_dir


def open_directory_no_follow(path: Path) -> int:
    absolute = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
        for index, part in enumerate(absolute.parts[1:], start=1):
            child = os.open(part, flags | no_follow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            validate_directory_metadata(os.fstat(descriptor), final=index == len(absolute.parts) - 1)
        return descriptor
    except (OSError, ValueError) as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ValueError("W&B output path cannot contain symlinks, untrusted directories, or special files") from exc


def validate_directory_metadata(metadata: os.stat_result, *, final: bool) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("W&B output path must contain only directories")
    if final and metadata.st_uid != os.geteuid():
        raise ValueError("W&B output directory must be owned by the current user")
    sticky = metadata.st_mode & stat.S_ISVTX
    world_writable = metadata.st_mode & stat.S_IWOTH
    group_writable = metadata.st_mode & stat.S_IWGRP
    root_owned = metadata.st_uid == 0
    if world_writable and not sticky:
        raise ValueError("W&B output path contains an untrusted writable directory")
    if group_writable and not sticky and not root_owned:
        raise ValueError("W&B output path contains an untrusted writable directory")
