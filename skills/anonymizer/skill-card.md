## Description: <br>
Produces a runnable Python script that calls the NeMo Anonymizer pipeline to anonymize text datasets via PII detection and entity replacement or LLM-powered rewriting. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers who need to anonymize text datasets containing personally identifiable information (PII) for privacy compliance, data sharing, or model training. <br>

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
- [Interactive Workflow Guide](references/interactive.md) <br>
- [NeMo Anonymizer Documentation](https://nvidia-nemo.github.io/Anonymizer/) <br>
- [GitHub Repository](https://github.com/NVIDIA-NeMo/Anonymizer.git) <br>
- [Choosing a Strategy](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/) <br>
- [Evaluation Guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) <br>


## Skill Output: <br>
**Output Type(s):** [Code] <br>
**Output Format:** [Python script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), each with 3 attempts per task in isolated sandbox pods. Dataset digest: sha256:2426de4eaba6137e3e0514953becf6a0eff19516d8d63bd790df3455ff8c8b32. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Measures whether the user's goal was achieved and expected workflow behavior was followed (equal-weight mean of goal_accuracy and behavior_check). <br>
- Efficiency: Evaluates tool-call productivity and token efficiency (50% skill_efficiency + 50% token_efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.0% | 94.0% |
| Security | 90.0% → 91.7% (+1.7 pts) | 93.8% → 100.0% (+6.2 pts) |
| Correctness | 50.0% → 93.3% (+43.3 pts) | 70.0% → 93.3% (+23.3 pts) |
| Discoverability | 96.3% | 93.8% |
| Effectiveness | 35.9% → 83.2% (+47.3 pts) | 42.7% → 86.9% (+44.2 pts) |
| Efficiency | 80.3% | 95.8% |

## Skill Version(s): <br>
441cdea (source: git SHA, committed 2026-09-17) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
