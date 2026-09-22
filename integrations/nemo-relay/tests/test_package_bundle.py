# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
PACKAGER = INTEGRATION_ROOT / "scripts/package_bundle.py"
ARTIFACT = Path("src/nemo_anonymizer_relay/worker.py")


def test_package_bundle_materializes_only_runtime_inputs(tmp_path: Path) -> None:
    output = tmp_path / "bundle"

    subprocess.run(
        [sys.executable, str(PACKAGER), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert {path.name for path in output.iterdir()} == {
        "LICENSE",
        "README.md",
        "config.schema.json",
        "pyproject.toml",
        "relay-plugin.toml",
        "src",
        "uv.lock",
    }
    assert not any(path.name == "__pycache__" for path in output.rglob("__pycache__"))
    assert not any(output.rglob("*.pyc"))
    assert {path.name for path in (output / "src/nemo_anonymizer_relay").iterdir()} == {
        "__init__.py",
        "transport.py",
        "worker.py",
    }
    assert (output / "LICENSE").read_bytes() == (REPOSITORY_ROOT / "LICENSE").read_bytes()

    artifact = output / ARTIFACT
    manifest = (output / "relay-plugin.toml").read_text(encoding="utf-8")
    assert artifact.is_file()
    assert "<artifact-sha256>" not in manifest
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() in manifest


def test_package_bundle_rejects_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "existing").write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(PACKAGER), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "bundle output must be an empty directory" in result.stderr
    assert (output / "existing").read_text(encoding="utf-8") == "keep"
