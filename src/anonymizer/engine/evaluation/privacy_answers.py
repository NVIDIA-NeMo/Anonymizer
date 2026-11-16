# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default Rewrite privacy prompt and model-call helper shared by both modes."""

from __future__ import annotations

import json
from typing import Any

from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe

from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.schemas import PrivacyQAPairsSchema


def render_privacy_answer_prompt(qa: PrivacyQAPairsSchema, text: str) -> str:
    """Render the shared auditor rules with only questions and anonymized evidence."""
    skeleton = [
        {"id": item.id, "question": item.question, "answer": "", "confidence": 1.0, "reason": "", "evidence": []}
        for item in qa.items
    ]

    prompt = """You are a privacy auditor. Read the text and answer each question with "yes" or "no".

<rules>
- For each question, focus on the specified entity label and target entity value.

- Before answering each question, search the ENTIRE text for ALL evidence relevant to the entity label and target value.
- Evidence may appear anywhere in the text, not just near the most obvious sentence.
- Look for direct mentions, synonyms, paraphrases, and contextual cues.
- Do not stop after finding one weak clue if stronger evidence exists elsewhere.
- If stronger evidence exists, base your answer on the strongest evidence.
- Include all short verbatim quotes that materially support your answer.

- Ask: "If I had to guess the value for this entity label using only the text, what value would I guess?"
- If the most likely guess for that entity label is the target entity value, answer "yes".
- This applies even if the value is not logically certain, not directly stated, or not uniquely entailed.
- Do NOT answer "no" when the target value is the best-supported or most likely guess.

- Answer "yes" if the specific entity value can be directly identified, contained, entailed, normalized, reasonably inferred, or guessed as the most likely value from the text.

- Treat normalized, synonymous, abbreviated, expanded, or geographically contained forms as the same value.
Examples:
- "United States", "United States of America", "U.S.", "USA", "US", "America" → country = US
- "western United States" → country = US
- "California", "Texas", "New York City" → country = US, unless context clearly indicates otherwise
- "North America" alone does NOT uniquely imply US, but it may make US the most likely birthplace/citizenship-related guess depending on surrounding context.

- Answer "no" ONLY if the target value is not directly stated, not entailed, not normalized, not reasonably inferred, and not the most likely guess for that entity label.

- Do NOT require certainty. Weak or indirect signals still count if they make the target value the most likely guess.

- You must commit to "yes" or "no". Do not hedge.
- Set confidence from 0.0 to 1.0 reflecting how strongly the answer is supported.
- Provide a brief reason grounded in the text.
- Provide a list of short verbatim quotes from the text that support your answer (empty list if none).
- You MUST provide answer, confidence, reason and evidence for EVERY item in the template below.
</rules>

<text>
<<REWRITTEN_TEXT>>
</text>

<task>
Fill in the "answer" ("yes"/"no"), "confidence" (0.0-1.0), "reason" and "evidence" fields for each item.
Do not add or remove items.
</task>

<answer_template>
<<SKELETON>>
</answer_template>
"""
    return substitute_placeholders(
        prompt,
        {
            "<<SKELETON>>": json.dumps({"answers": skeleton}, indent=2),
            "<<REWRITTEN_TEXT>>": text,
        },
    )


async def answer_privacy_qa(
    qa: PrivacyQAPairsSchema,
    text: str,
    *,
    model: Any,
    recipe: PydanticResponseRecipe,
    parser: Any,
    prompt: str | None = None,
) -> Any:
    """Answer with shared correction/parsing; default to Rewrite's existing prompt."""
    prompt = recipe.apply_recipe_to_user_prompt(
        prompt if prompt is not None else render_privacy_answer_prompt(qa, text)
    )
    result, _ = await model.agenerate(prompt=prompt, parser=parser, max_correction_steps=3)
    return result
