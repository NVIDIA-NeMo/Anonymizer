# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route-neutral execution harness shared by the detection, replace and rewrite adapters.

``frames`` holds the frame-publication values the Graph SDK dispatches on; ``kernel`` holds
plan preparation, admission/start, synchronous hosting and projection used by every route.
"""
