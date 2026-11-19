<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ideas: what you can build with the graph SDK

After reading this page you will have a set of workflow patterns to start from, each built only from the public
`anonymizer.graph` names. Two are shown as complete programs; the rest name the pieces and the guide to read.

!!! warning "Provisional"

    The graph SDK is provisional. Names, signatures and default values may change in a later release.

## A rewrite with a bounded repair loop

A model drafts a rewrite, a quick check inside the step asks for a repair, and the step gives up after a fixed number
of tries. A separate verifier, not the step's own check, decides release. The step declares the most requests it can
make, so `prepare` can refuse it against a budget.

The stand-in model below needs one repair for one of the texts. Replace `StandInModel` with your client.

```python
from anonymizer import graph as g

MAX_TRIES = 3


class StandInModel:
    """Stands in for a model: the first draft of a text keeps the surname, a repair removes it."""

    def __init__(self) -> None:
        self.requests = 0

    def rewrite(self, texts: list[str], feedback: list[str]) -> list[str]:
        self.requests += 1
        out = []
        for text, note in zip(texts, feedback):
            text = text.replace("Dana", "the customer")
            out.append(text.replace(" Reyes", "") if note else text)
        return out


model = StandInModel()


def leaks(text: str) -> bool:
    return "Dana" in text or "Reyes" in text


def rewrite_with_repair(source: tuple[str, ...]) -> list[str | Exception]:
    drafts = model.rewrite(list(source), [""] * len(source))
    for _ in range(MAX_TRIES - 1):
        todo = [i for i, d in enumerate(drafts) if leaks(d)]
        if not todo:
            break
        fixed = model.rewrite([source[i] for i in todo], ["remove the name"] * len(todo))
        for i, d in zip(todo, fixed):
            drafts[i] = d
    return [ValueError("still leaks after repairs") if leaks(d) else d for d in drafts]


def final_check(text: str, target: g.Datum) -> g.Assessment:
    return g.Assessment(not leaks(text), target, "name_present")


scope = g.GraphScope()
targets = (scope.datum("Dana asked about a refund."), scope.datum("Dana Reyes called twice."))
workflow = g.Workflow(
    "rewrite",
    (
        g.Operation.phase_scoped(
            "rewrite", rewrite_with_repair, {"source": "text"}, requests=g.Requests.at_most(MAX_TRIES)
        ),
        g.Operation.verifier(
            "final-check", final_check, {"text": "text", "target": "datum"}, checks=("no_name",), subject=("text",)
        ),
    ),
    requires=("no_name",),
)
prepared = g.prepare(g.ProtectionGraph(scope, targets), workflow, request_budget=MAX_TRIES)
result = g.protect(prepared)
print(prepared.summary().planned_requests, model.requests)
print([result.protected.get(t) for t in targets])
```

```text
upper 3 2
['the customer asked about a refund.', 'the customer called twice.']
```

The plan allows at most three requests; the run used two (one draft call for both texts, one repair call for the
second). The step returns an `Exception` for a text that still leaks after the last try, which fails only that
target. Even when the step's own check is satisfied, release waits for `final-check`.

## Audit text that was anonymized elsewhere

An annotation workflow runs verifiers over existing output and reports what they found, without releasing anything.
Use it to measure leftover identifiers in a dataset produced by another tool.

```python
from anonymizer import graph as g

already_anonymized = ["[NAME] moved to [CITY].", "Call Dana at [PHONE].", "No personal data."]

scope = g.GraphScope()
rows = tuple(scope.datum(t) for t in already_anonymized)
check = g.Operation.verifier(
    "leftover-name",
    lambda text, target: g.Assessment("Dana" not in text, target, "name_present"),
    {"text": "text", "target": "datum"},
    checks=("no_leftover_name",),
    subject=("text",),
)
workflow = g.Workflow("audit", (check,), release="annotation", requires=("no_leftover_name",))
result = g.protect(g.prepare(g.ProtectionGraph(scope, rows), workflow))
print([(o.qualification, o.reason) for o in result.outcomes])
print(result.summary().qualifications, len(result.protected))
```

```text
[('met', 'annotation'), ('not_met', 'failed'), ('met', 'annotation')]
(('met', 2), ('not_met', 1)) 0
```

The verifier reads the `text` seed, which with no transform is the original text. Nothing is released
(`len(result.protected)` is 0); the qualification counts are the audit result.

## More patterns

Each pattern below uses only public names. None is shown as a full program here.

| Pattern | How | Pieces | Guide |
|---|---|---|---|
| Bring your own detector | A phase-scoped step calls a local NER model or a service once for all targets and returns mentions; a later step replaces them; a separate verifier rescans the output. | `Operation.phase_scoped`, `Mention`, `Operation.verifier` | [Write your own detector](guide-custom-detector.md) |
| Release a row only when every field passes | Each field is a target; the row's fields form one output group; a derived field depends on its source fields. | `OutputGroup`, `Dependency`, `Record` | [Release related texts together](guide-release-together.md) |
| One prompt per row | A phase-scoped step reads the `record` seed and sends the fields of each record to the model together. | `Record`, `Operation.phase_scoped` | [Release related texts together](guide-release-together.md#records) |
| Conversation-aware protection | Earlier turns, or a structured profile, reach a step as bounded context that is read but never released. | `ContextScope`, `ContextSource`, `ContextView`, `register_kind`, `Limits(max_context_bytes=)` | [Give an operation context](guide-give-an-operation-context.md) |
| Long documents by section | Each section is a target, so a failure stays in its section; the document is rebuilt only from available candidates. | `ProtectionResult.assemble_regions` | [Rebuild a document](guide-rebuild-documents.md) |
| Consistent keys across a trace | Tool results, messages and tool arguments of one agent trace are targets; a phase-scoped step assigns one substitute per real value across all of them; an output group releases the trace as a whole; a verifier rejects two values mapped to one key. | `Operation.phase_scoped`, `OutputGroup`, `Operation.verifier` | [Process all targets in one call](guide-batch-targets.md) |
| Compare two strategies on the same data | Build two workflows over the same graph and compare their summaries; `prepare` and `protect` keep no state between runs. | `prepare`, `protect`, `InspectionSummary` | [Find out why a target was withheld](guide-read-outcomes.md) |

## Limits of what the SDK offers today

- A step cannot report how many requests it actually made through a public name, so observed request counts stay
  `unknown` for steps that declare requests.
- Nothing is kept between calls to `protect`. Reusing the same substitute for the same value across two runs needs
  your own store.
- Steps run one at a time on the calling thread. Concurrency, if you need it, goes inside a phase-scoped step.
