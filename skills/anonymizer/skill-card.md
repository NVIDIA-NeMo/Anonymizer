## Description: <br>
Use when the user wants to anonymize a text dataset, redact PII, de-identify free-text data, or rewrite text to remove sensitive or inferable identifying information. Produces a runnable Python script that calls the NeMo Anonymizer pipeline (detection → replace or rewrite). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers use this skill to detect and protect PII in a text dataset, generating a runnable Python script that configures and runs the NeMo Anonymizer pipeline (detection followed by replace or rewrite), previews results, inspects failures and quality metrics, and optionally scores output with LLM-as-judge evaluation. <br>

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
- [Detection](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/detection/) <br>
- [Evaluation](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) <br>
- [Models](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/) <br>
- [Self-hosting GLiNER2](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/) <br>
- [Troubleshooting](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Runnable Python script (Markdown fenced code during interaction)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), 1 attempt per task, each run in its own isolated k8s-sandbox pod. Dataset snapshot: skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.0% (uplift unavailable) | 93.2% (uplift unavailable) |
| Security | 100.0% → 83.3% (-16.7 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 56.7% → 86.7% (+30.0 points) | 76.7% → 100.0% (+23.3 points) |
| Discoverability | 99.5% (uplift unavailable) | 93.8% (uplift unavailable) |
| Effectiveness | 56.3% → 89.2% (+32.9 points) | 59.9% → 91.8% (+31.9 points) |
| Efficiency | 81.5% (uplift unavailable) | 97.0% (uplift unavailable) |

## Skill Version(s): <br>
188ef36 (source: git SHA, committed 2026-10-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
