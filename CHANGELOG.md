<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release notes

## Unreleased

- The bundled GLiNER2 server uses overlapping native word windows with defaults of 128 tokens and 32-token overlap, and grounds predictions against source chunks before document remapping.
- Detector model YAML `inference_parameters.extra_body.chunk_length` and `overlap` are now passed through when configured and omitted when absent, allowing the server to choose defaults. Previously the engine always overwrote them with 384/128. The shipped localhost model entry omits both fields; explicitly set values in custom model YAML if another service needs the old request parameters (whose units are server-specific).
- The 128/32 chunk defaults refer to native word tokens in the updated bundled server, not characters or encoder subwords. Chunk units are defined by each server; updating client YAML alone does not change the units of an older or separately hosted server.
