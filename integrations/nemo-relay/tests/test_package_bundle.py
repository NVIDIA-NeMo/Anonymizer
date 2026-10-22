# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
PACKAGER = INTEGRATION_ROOT / "scripts/package_bundle.py"
ARTIFACT = Path("nemo_anonymizer_relay/worker.py")


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
        "nemo_anonymizer_relay",
        "pyproject.toml",
        "relay-plugin.toml",
    }
    assert not any(path.name == "__pycache__" for path in output.rglob("__pycache__"))
    assert not any(output.rglob("*.pyc"))
    assert {path.name for path in (output / "nemo_anonymizer_relay").iterdir()} == {
        "__init__.py",
        "backend.py",
        "projection.py",
        "sanitizer.py",
        "worker.py",
    }
    assert (output / "LICENSE").read_bytes() == (REPOSITORY_ROOT / "LICENSE").read_bytes()

    artifact = output / ARTIFACT
    manifest_text = (output / "relay-plugin.toml").read_text(encoding="utf-8")
    manifest = tomllib.loads(manifest_text)
    entrypoint_module = manifest["load"]["entrypoint"].split(":", 1)[0]
    entrypoint_artifact = output.joinpath(*entrypoint_module.split(".")).with_suffix(".py")
    assert artifact.is_file()
    assert entrypoint_artifact.samefile(artifact)
    assert "<artifact-sha256>" not in manifest_text
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() in manifest_text
    bundled_pyproject = (output / "pyproject.toml").read_text(encoding="utf-8")
    assert "nemo-anonymizer>0.4.0,<1" in bundled_pyproject
    assert "[tool.uv.sources]" not in bundled_pyproject


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
