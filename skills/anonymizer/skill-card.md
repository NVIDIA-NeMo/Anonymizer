## Description: <br>
Produces a runnable Python script that calls the NeMo Anonymizer pipeline (detection → replace or rewrite) to anonymize text datasets, redact PII, de-identify free-text data, or rewrite text to remove sensitive or inferable identifying information. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers who need to anonymize text datasets containing PII, using entity detection with replacement or LLM-powered rewriting to produce privacy-safe data for downstream use. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Interactive workflow guide](references/interactive.md) <br>
- [NeMo Anonymizer documentation](https://nvidia-nemo.github.io/Anonymizer/) <br>
- [GitHub repository](https://github.com/NVIDIA-NeMo/Anonymizer) <br>
- [Choosing a strategy](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/) <br>
- [Evaluation concepts](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) <br>
- [Self-hosting GLiNER2](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/) <br>


## Skill Output: <br>
**Output Type(s):** [Code] <br>
**Output Format:** [Python script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative) with 3 attempts each, run in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use, including unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed, including skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal, combining goal completion and expected workflow adherence. <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and token usage, combining tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.1% — uplift unavailable | 88.8% — uplift unavailable |
| Security | 100.0% → 91.7% (-8.3 pts) | 100.0% → 83.3% (-16.7 pts) |
| Correctness | 57.8% → 86.7% (+28.9 pts) | 70.0% → 100.0% (+30.0 pts) |
| Discoverability | 97.5% — uplift unavailable | 92.5% — uplift unavailable |
| Effectiveness | 39.8% → 89.3% (+49.5 pts) | 51.6% → 93.0% (+41.4 pts) |
| Efficiency | 80.4% — uplift unavailable | 75.2% — uplift unavailable |

## Skill Version(s): <br>
f7d89e7 (source: git SHA, committed 2026-09-16) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
