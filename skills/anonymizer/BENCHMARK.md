# Skill Benchmark: anonymizer

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `anonymizer`
- Evaluation date: 2026-10-07
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:c2c13b2d794c6117dac0402f1261bd2d80972085c5e716426bedff6b3d59b8ae` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
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
| Overall | 89.4% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 83.3% (-16.7 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 53.3% → 96.7% (+43.4 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.4% → 86.6% (+31.2 points) | 62.2% → 88.3% (+26.1 points) |
| Efficiency | 80.4% — baseline ran, but no comparable score was available; uplift unavailable | 86.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,277,114 | 975,513 | +301,601 | +30.92% | skill 6/6; base 6/6 |
| claude-code | anonymizer-negative-general-privacy-explainer | 30,315 | 30,878 | -563 | -1.82% | skill 1/1; base 1/1 |
| claude-code | anonymizer-negative-repository-source-development | 247,738 | 219,194 | +28,544 | +13.02% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-failed-records-first | 225,452 | 184,016 | +41,436 | +22.52% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-hash-cross-record-consistency | 188,846 | 31,663 | +157,183 | +496.42% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-mode-choice | 153,059 | 33,134 | +119,925 | +361.94% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-self-hosted-gliner | 431,704 | 476,628 | -44,924 | -9.43% | skill 1/1; base 1/1 |
| codex | All cases | 643,364 | 677,093 | -33,729 | -4.98% | skill 6/6; base 6/6 |
| codex | anonymizer-negative-general-privacy-explainer | 13,761 | 13,626 | +135 | +0.99% | skill 1/1; base 1/1 |
| codex | anonymizer-negative-repository-source-development | 435,709 | 540,694 | -104,985 | -19.42% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-failed-records-first | 82,924 | 41,198 | +41,726 | +101.28% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-hash-cross-record-consistency | 29,789 | 17,691 | +12,098 | +68.39% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-mode-choice | 30,256 | 24,597 | +5,659 | +23.01% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-self-hosted-gliner | 50,925 | 39,287 | +11,638 | +29.62% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,920,478 | 1,652,606 | +267,872 | +16.21% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/anonymizer/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (5445 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/anonymizer/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in interactive.md (`skills/anonymizer/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/anonymizer/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/anonymizer/SKILL.md`)
- 9 additional finding(s) are available in the full evaluation artifacts.

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
