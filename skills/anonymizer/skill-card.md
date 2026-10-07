## Description: <br>
Produces a runnable Python script that calls the NeMo Anonymizer pipeline to detect and protect PII through context-aware entity replacement or LLM-powered rewriting. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers who need to anonymize text datasets by detecting and replacing PII or rewriting text to remove sensitive and inferable identifying information. <br>

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
- [Choosing a Strategy](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/) <br>
- [Detection](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/detection/) <br>
- [Evaluation](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) <br>
- [Models](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/) <br>
- [Self-hosting GLiNER](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/) <br>
- [Troubleshooting](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/) <br>
- [GitHub Repository](https://github.com/NVIDIA-NeMo/Anonymizer.git) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files] <br>
**Output Format:** [Python script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative) in isolated sandbox pods, evaluated by skill-evaluator v1.5.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and followed the expected workflow behavior. <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and excessive token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.4% | 90.4% |
| Security | 100.0% → 83.3% (-16.7 pts) | 83.3% → 83.3% (±0.0 pts) |
| Correctness | 53.3% → 96.7% (+43.4 pts) | 90.0% → 100.0% (+10.0 pts) |
| Discoverability | 100.0% | 93.8% |
| Effectiveness | 55.4% → 86.6% (+31.2 pts) | 62.2% → 88.3% (+26.1 pts) |
| Efficiency | 80.4% | 86.4% |

## Skill Version(s): <br>
87006d9 (source: git SHA, committed 2026-10-07) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
