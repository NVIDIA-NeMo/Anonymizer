## Description: <br>
Use when the user wants to anonymize a text dataset, redact PII, de-identify free-text data, or rewrite text to remove sensitive or inferable identifying information. Produces a runnable Python script that calls the NeMo Anonymizer pipeline (detection → replace or rewrite). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to anonymize text datasets by detecting and protecting PII, generating a runnable Python script that calls the NeMo Anonymizer pipeline in replace or rewrite mode. <br>

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
- [Interactive Workflow](references/interactive.md) <br>
- [Choosing a strategy](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/) <br>
- [Detection guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/detection/) <br>
- [Evaluation guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) <br>
- [Models guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/) <br>
- [Self-hosting GLiNER](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/) <br>
- [Troubleshooting guide](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/) <br>
- [NeMo Anonymizer repository](https://github.com/NVIDIA-NeMo/Anonymizer.git) <br>


## Skill Output: <br>
**Output Type(s):** [Code] <br>
**Output Format:** [Runnable Python script (Markdown-wrapped code block)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), 1 attempt per task, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Combines tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 85.3% — uplift unavailable | 87.2% — uplift unavailable |
| Security | 83.3% → 83.3% (±0.0 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 73.3% → 86.7% (+13.4 points) | 76.7% → 93.3% (+16.6 points) |
| Discoverability | 100.0% — uplift unavailable | 92.5% — uplift unavailable |
| Effectiveness | 53.8% → 80.3% (+26.5 points) | 52.5% → 78.7% (+26.2 points) |
| Efficiency | 76.0% — uplift unavailable | 87.9% — uplift unavailable |

## Skill Version(s): <br>
dc4f699 (source: git SHA, committed 2026-10-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
