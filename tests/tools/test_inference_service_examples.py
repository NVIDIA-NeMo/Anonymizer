# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example role configuration must route to the shipped serving profiles."""

from __future__ import annotations

from pathlib import Path

import httpx

from anonymizer.engine.ndd import model_loader
from inference_service_compiler.profiles import load_profile

PROFILES = Path(__file__).resolve().parents[2] / "tools" / "inference_service_profiles"


def test_example_role_configuration_routes_each_model_to_the_profile_serving_it() -> None:
    examples = PROFILES / "examples"
    parsed = model_loader.parse_model_configs(examples / "local-models.yaml")
    providers = model_loader.parse_model_providers(examples / "local-providers.yaml")
    model_loader.validate_model_configs_reference_providers(parsed.model_configs, providers)
    served_by_port = {spec.port: spec.backend.model_name for spec in map(load_profile, PROFILES.glob("*.toml"))}
    endpoint_by_provider = {provider.name: provider.endpoint for provider in providers}
    for config in parsed.model_configs:
        port = httpx.URL(endpoint_by_provider[config.provider]).port
        assert port is not None, config.provider
        assert served_by_port[port] == config.model, config.alias
