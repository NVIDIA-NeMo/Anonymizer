# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_package_imports() -> None:
    """Verify the anonymizer package and subpackages are importable."""
    import anonymizer
    import anonymizer.config
    import anonymizer.engine
    import anonymizer.interface

    assert anonymizer.__version__ == "0.1.0"
