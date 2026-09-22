# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the NeMo Relay worker bundle owned by this integration."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
PACKAGE_PATH = Path("src/nemo_anonymizer_relay")
ARTIFACT_PATH = PACKAGE_PATH / "worker.py"
WORKER_SOURCES = (
    PACKAGE_PATH / "__init__.py",
    PACKAGE_PATH / "transport.py",
    ARTIFACT_PATH,
)
HASH_PLACEHOLDER = "<artifact-sha256>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_bundle(output: Path) -> Path:
    """Copy the minimal installable worker bundle into an empty directory."""

    required = [
        INTEGRATION_ROOT / "README.md",
        INTEGRATION_ROOT / "config.schema.json",
        INTEGRATION_ROOT / "pyproject.toml",
        INTEGRATION_ROOT / "relay-plugin.toml",
        INTEGRATION_ROOT / "uv.lock",
        *(INTEGRATION_ROOT / path for path in WORKER_SOURCES),
        REPOSITORY_ROOT / "LICENSE",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"required bundle input is missing: {', '.join(missing)}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"bundle output must be an empty directory: {output}")

    manifest = (INTEGRATION_ROOT / "relay-plugin.toml").read_text(encoding="utf-8")
    if manifest.count(HASH_PLACEHOLDER) != 1:
        raise ValueError("relay-plugin.toml must contain one artifact hash placeholder")

    output.mkdir(parents=True, exist_ok=True)
    (output / PACKAGE_PATH).mkdir(parents=True)
    for path in WORKER_SOURCES:
        shutil.copy2(INTEGRATION_ROOT / path, output / path)
    for name in ("README.md", "config.schema.json", "pyproject.toml", "uv.lock"):
        shutil.copy2(INTEGRATION_ROOT / name, output / name)
    shutil.copy2(REPOSITORY_ROOT / "LICENSE", output / "LICENSE")

    artifact_digest = _sha256(output / ARTIFACT_PATH)
    (output / "relay-plugin.toml").write_text(
        manifest.replace(HASH_PLACEHOLDER, artifact_digest),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(package_bundle(args.output.resolve()))


if __name__ == "__main__":
    main()
