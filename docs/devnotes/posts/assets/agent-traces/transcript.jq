# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# OpenAI messages -> one "<|role|> body" transcript per trace (system prompt dropped), as CSV; skips traces over 10k tokens
(["trace_id", "text"] | @csv),
(.[] | select(.metadata.stats.tokens | (.user + .assistant + .tool) <= 10000)
 | [.metadata.record_id,
    ([.messages[] | select(.role != "system")
      | (.content // "" | gsub("^\\s+|\\s+$"; "")) as $body
      | if .role == "tool" then "<|tool_result|> \($body)"
        else (if $body != "" then "<|\(.role)|> \($body)" else empty end),
             (.tool_calls[]? | "<|tool_call:\(.function.name)|> \(.function.arguments)")
        end] | join("\n\n"))]
 | @csv)
