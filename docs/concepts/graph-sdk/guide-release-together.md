<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release related texts together

After reading this page you can release the fields of one table row all together or not at all, withhold a
derived text when its source is withheld, and find out which target caused a withholding.

## Three relations

| Relation | Use it when | Changes release? |
|---|---|---|
| `OutputGroup(members)` | Several targets must be released together or not at all, such as the fields of one row. | Yes |
| `Dependency(dependent, on)` | One target is derived from another, such as a summary of a note; if the note is withheld, the summary must be too. | Yes |
| `Record` (from `scope.record(members)`) | Several targets belong together and steps should know it, such as to put one row in one prompt. | No |

Groups may not overlap, dependencies may not form a cycle (`group_overlap`, `dependency_cycle`), and each target is
in at most one declared record (`record_overlap`).

## A row released as a unit

Three fields of one row are one output group and one record. A summary depends on the `note` field. The redaction
step misses "Dana" in the note, so the note fails its check:

```python
from anonymizer import graph as g

row = {"name": "Dana Reyes", "city": "Oakland", "note": "Dana asked about her refund."}

scope = g.GraphScope()
fields = {key: scope.datum(value) for key, value in row.items()}
summary = scope.datum("Customer asked about a refund.")
record = scope.record(tuple(fields.values()))

graph = g.ProtectionGraph(
    scope,
    (*fields.values(), summary),
    groups=(g.OutputGroup(tuple(fields.values())),),
    dependencies=(g.Dependency(summary, fields["note"]),),
    records=(record,),
)


def redact(text: str) -> str:
    return text.replace("Dana Reyes", "[NAME]").replace("Oakland", "[CITY]")


def no_name(text: str, target: g.Datum) -> g.Assessment:
    return g.Assessment("Dana" not in text, target)


workflow = g.Workflow(
    "row",
    (
        g.Operation("redact", redact),
        g.Operation.verifier(
            "check", no_name, {"text": "text", "target": "datum"}, checks=("no_name",), subject=("text",)
        ),
    ),
    requires=("no_name",),
)
result = g.protect(g.prepare(graph, workflow))
for key, datum in (*fields.items(), ("summary", summary)):
    o = result.outcome(datum)
    by = [k for k, d in fields.items() if d in o.withheld_by]
    print(f"{key:8} {o.qualification:8} protected={o.protected} reason={o.reason} withheld_by={by}")
```

```text
name     met      protected=False reason=withheld withheld_by=['note']
city     met      protected=False reason=withheld withheld_by=['note']
note     not_met  protected=False reason=failed withheld_by=[]
summary  met      protected=False reason=withheld withheld_by=['note']
```

`name` and `city` passed their own check but are withheld because their group member `note` was. `summary` passed
too, but depends on `note`. `withheld_by` names the targets that caused the withholding, so you can trace each
withheld target back to the one that actually failed.

Release is decided after all checks: a target is released only when it is `met` and nothing it depends on, and no
member of its group, is withheld. Withholding spreads along dependencies and through groups until nothing changes.

## Records

A record does not change release. It tells steps which targets belong together. `graph.record_of(target)` returns
a target's record; a target in no declared record is its own one-member record. `graph.all_records` lists every
record once.

A step that reads the `record` seed receives the target's `Record`. In a phase-scoped step this lets one call
group its batch by row:

```python
from anonymizer import graph as g

scope = g.GraphScope()
row_a = (scope.datum("Dana Reyes"), scope.datum("Oakland"))
row_b = (scope.datum("Sam Lee"),)
graph = g.ProtectionGraph(scope, (*row_a, *row_b), records=(scope.record(row_a),))


def label_by_row(text: tuple[str, ...], record: tuple[g.Record, ...]) -> list[str]:
    ids: dict[g.Record, int] = {}
    return [f"row{ids.setdefault(r, len(ids))}: {t}" for t, r in zip(text, record)]


step = g.Operation.phase_scoped("label", label_by_row, {"text": "text", "record": "record"})
result = g.protect(g.prepare(graph, g.Workflow("label", (step,))))
print([text for _, text in result.content()])
print(len(graph.all_records))
```

```text
['row0: Dana Reyes', 'row0: Oakland', 'row1: Sam Lee']
2
```

Records compare by identity, never by their members' text. A record's `repr` shows only its member count.
