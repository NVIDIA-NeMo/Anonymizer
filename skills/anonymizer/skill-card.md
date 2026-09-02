## Description: <br>
Use when the user wants to anonymize a text dataset, redact PII, de-identify free-text data, or rewrite text to remove sensitive or inferable identifying information. Produces a runnable Python script that calls the NeMo Anonymizer pipeline (detection → replace or rewrite). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers who need to anonymize text datasets containing PII for downstream model training, analytics, or data sharing. <br>

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
- [Interactive workflow reference](references/interactive.md) <br>
- [NeMo Anonymizer documentation](https://nvidia-nemo.github.io/Anonymizer/) <br>
- [Choosing a strategy](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/) <br>
- [Detection guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/detection/) <br>
- [Evaluation guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) <br>
- [Models guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/) <br>
- [Troubleshooting](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/) <br>
- [Self-hosting GLiNER](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/) <br>
- [GitHub repository](https://github.com/NVIDIA-NeMo/Anonymizer.git) <br>


## Skill Output: <br>
**Output Type(s):** [Code] <br>
**Output Format:** [Python script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Runnable script with CLI flags for preview, full run, and evaluation] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative) from skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill is found and executed when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal (goal completion + expected workflow adherence). <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage (routing quality and productive tool use). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 65% → 95% (+30 points) | 72% → 93% (+21 points) |
| Security | 100% → 100% (±0 points) | 83% → 83% (±0 points) |
| Correctness | 73% → 97% (+23 points) | 90% → 100% (+10 points) |
| Discoverability | 50% → 98% (+48 points) | 67% → 92% (+25 points) |
| Effectiveness | 54% → 87% (+33 points) | 69% → 92% (+23 points) |
| Efficiency | 50% → 94% (+44 points) | 50% → 100% (+50 points) |

## Skill Version(s): <br>
318cc15 (source: git SHA, committed 2026-08-19) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
