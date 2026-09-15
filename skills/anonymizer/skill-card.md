## Description: <br>
Anonymizes text datasets by detecting PII and applying replace or rewrite strategies via the NeMo Anonymizer pipeline, producing a runnable Python script as output. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers who need to anonymize text datasets by detecting and replacing or rewriting personally identifiable information (PII) for privacy compliance, data sharing, or model training. <br>

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
- [Choosing a strategy](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/) <br>
- [Detection guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/detection/) <br>
- [Evaluation guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) <br>
- [Models guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/) <br>
- [Self-hosting GLiNER](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/) <br>
- [Troubleshooting](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/) <br>
- [GitHub repository](https://github.com/NVIDIA-NeMo/Anonymizer.git) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files] <br>
**Output Format:** [Python script (.py)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative) with 3 attempts each, run in isolated sandbox pods. Dataset digest: sha256:c2c13b2d794c6117dac0402f1261bd2d80972085c5e716426bedff6b3d59b8ae. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Checks tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.7% | 93.7% |
| Security | 95.0% → 100.0% (+5.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 34.0% → 83.3% (+49.3 points) | 75.0% → 96.7% (+21.7 points) |
| Discoverability | 99.5% | 95.0% |
| Effectiveness | 35.0% → 83.0% (+48.0 points) | 50.8% → 89.8% (+39.0 points) |
| Efficiency | 82.7% | 86.9% |

## Skill Version(s): <br>
b381e46 (source: git SHA, committed 2026-09-15) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
