# Skill Benchmark: anonymizer

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `anonymizer`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:2426de4eaba6137e3e0514953becf6a0eff19516d8d63bd790df3455ff8c8b32` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 90.0% → 91.7% (+1.7 points) | 93.8% → 100.0% (+6.2 points) |
| Correctness | 50.0% → 93.3% (+43.3 points) | 70.0% → 93.3% (+23.3 points) |
| Discoverability | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 35.9% → 83.2% (+47.3 points) | 42.7% → 86.9% (+44.2 points) |
| Efficiency | 80.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,300,167 | 3,301,368 | N/A | N/A | skill 6/6; base 10/10 |
| claude-code | anonymizer-negative-general-privacy-explainer | 30,496 | 30,269 | +227 | +0.75% | skill 1/1; base 1/1 |
| claude-code | anonymizer-negative-repository-source-development | 186,217 | 151,863 | +34,354 | +22.62% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-failed-records-first | 300,419 | 336,821 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | anonymizer-positive-hash-cross-record-consistency | 317,920 | 32,581 | +285,339 | +875.78% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-mode-choice | 277,677 | 32,748 | +244,929 | +747.92% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-self-hosted-gliner | 1,187,438 | 2,717,086 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 644,464 | 568,153 | N/A | N/A | skill 6/6; base 8/8 |
| codex | anonymizer-negative-general-privacy-explainer | 13,746 | 13,523 | +223 | +1.65% | skill 1/1; base 1/1 |
| codex | anonymizer-negative-repository-source-development | 392,086 | 358,548 | +33,538 | +9.35% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-failed-records-first | 30,171 | 123,207 | N/A | N/A | skill 1/1; base 3/3 |
| codex | anonymizer-positive-hash-cross-record-consistency | 29,987 | 17,746 | +12,241 | +68.98% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-mode-choice | 30,319 | 22,622 | +7,697 | +34.02% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-self-hosted-gliner | 148,155 | 32,507 | +115,648 | +355.76% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,944,631 | 3,869,521 | N/A | N/A | skill 12/12; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/anonymizer/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in interactive.md (`skills/anonymizer/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/anonymizer/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/anonymizer/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (274 chars, recommend 50-150) (`skills/anonymizer/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
