# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from anonymizer.config.anonymizer_config import DetectionScope
from anonymizer.config.replace_strategies import Annotate, ReplaceMethod
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_SENSITIVITY_DISPOSITION
from anonymizer.engine.graph_core.execution import TargetOutcome
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.rewrite.rewrite_workflow import rewrite_leftovers
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.engine.value_matching import Leftover, count_values, entity_spans
from anonymizer.interface.display import render_record_html


class _DisplayMixin:
    """Shared ``display_record`` behavior for result types."""

    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    _display_cycle_index: int
    _outcomes: tuple[TargetOutcome, ...] | None

    def outcome(self, row: int) -> TargetOutcome:
        """The protection outcome of output row ``row``: its ``qualification`` and the ``reasons`` behind it.

        ``qualification`` is ``"met"`` when every required check passed for the row, ``"not_met"`` when the
        row did not finish or a required check failed, could not decide (a ``could_not_assess`` reason) or gave
        no assessment, and ``"not_assessed"`` only when no check is required.

        Raises:
            IndexError: ``row`` is not an output row, or this result carries no outcomes (an ``evaluate()``,
                custom runner's or hand-built result).
        """
        outcomes = getattr(self, "_outcomes", None)  # absent on results pickled before outcomes were kept
        if outcomes is None:
            raise IndexError("This result carries no protection outcomes.")
        if row < 0 or row >= len(outcomes):
            raise IndexError(f"Record index {row} is out of bounds for {len(outcomes)} records.")
        return outcomes[row]

    def leftovers(self, row: int) -> tuple[Leftover, ...]:
        """Each detected value of output row ``row`` as the leftover check counts it, in ``cause`` order: Replace's
        advisory ``no_detected_value_added``, or Rewrite's ``replacements_applied`` and its review flag. Holds record
        text, for auditing your own result.

        Raises:
            IndexError: ``row`` is not a row of ``trace_dataframe``.
            ValueError: an Annotate result, or one without the source, output or ``final_entities`` columns.
        """
        if isinstance(getattr(self, "replace_method", None), Annotate):
            raise ValueError("Annotate results have no leftover check.")
        frame, text = self.trace_dataframe, self.resolved_text_column
        rewritten = f"{text}_rewritten"
        output = rewritten if rewritten in frame.columns else f"{text}_replaced"
        if not {text, output, COL_FINAL_ENTITIES} <= set(frame.columns):
            raise ValueError("This result has no leftover trace (source, output and final_entities columns).")
        if row < 0 or row >= len(frame):
            raise IndexError(f"Record index {row} is out of bounds for {len(frame)} records.")
        record = frame.iloc[row].to_dict()
        source = record[text] if isinstance(record[text], str) else ""
        if output == rewritten:
            disposition = record.get(COL_SENSITIVITY_DISPOSITION)
            return rewrite_leftovers(source, record[COL_FINAL_ENTITIES], record[output], disposition) or ()
        # The Replace verifier counts the row's mentions; the graph admits only exact, non-overlapping mentions of
        # these entities, so converting final_entities here gives the same spans.
        entities = EntitiesSchema.from_raw(record[COL_FINAL_ENTITIES]).entities
        return count_values(source, entity_spans(source, entities), str(record[output]))

    def display_record(self, index: int | None = None) -> None:
        """Render a record with entity highlights and replacement map in a notebook.

        Args:
            index: Row index to display. If None, cycles through records on repeated calls.
        """
        i = index if index is not None else self._display_cycle_index
        if i < 0 or i >= len(self.trace_dataframe):
            raise IndexError(f"Record index {i} is out of bounds for {len(self.trace_dataframe)} records.")

        row = self.trace_dataframe.iloc[i]
        html_str = render_record_html(row, record_index=i, resolved_text_column=self.resolved_text_column)

        try:
            from IPython.display import HTML, display

            display(HTML(html_str))
        except ImportError:
            print(html_str)

        if index is None:
            self._display_cycle_index = (self._display_cycle_index + 1) % len(self.trace_dataframe)


@dataclass
class AnonymizerResult(_DisplayMixin):
    """Result returned by full anonymization runs.

    Attributes:
        dataframe: User-facing columns only (text, replaced/rewritten text, scores).
        trace_dataframe: Full pipeline trace including all internal columns.
        resolved_text_column: Name of the user-facing text column. Equals the
            user's requested ``text_column`` unless the reader had to rename it
            to avoid colliding with an Anonymizer output column, in which case
            it is the post-rename identifier (e.g. ``"final_entities__input"``).
        failed_records: Records that failed during pipeline processing.
        replace_method: The replace strategy that produced this result. Set by
            ``run()`` / ``preview()``; consumed by ``evaluate()`` to dispatch the
            right judges. ``None`` on results that were constructed by hand or
            loaded from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this result when rewrite
            mode was used. Set by ``run()`` / ``preview()``; consumed by
            ``evaluate()`` to dispatch the rewrite judges. Mutually exclusive
            with ``replace_method``.
        entity_labels: Allowlist of entity labels that were in scope during
            detection. Preserved for ``evaluate()`` so the coverage judge scopes
            its evaluation to the same label set. ``None`` means detection was
            permissive: default labels were configured, and the augmenter may
            have emitted additional labels. Configured examples are intentionally
            not persisted for evaluation.
        data_summary: Optional dataset context supplied with the original input.
            Preserved for ``evaluate()`` so entity-coverage judging uses the
            same context as detection.
        excluded_entity_labels: Labels that were explicitly excluded from
            detection. Preserved for ``evaluate()`` so the coverage judge does
            not penalise the output for not anonymizing excluded labels.
        id_column: Record identifier column retained in ``dataframe`` for
            in-memory record input. File-backed input currently leaves this as
            ``None``.
        detection_scope: The resolved detection scope the run used: effective
            labels, allowed labels (``None`` when ``entity_labels`` is unset),
            exclusions, and GLiNER threshold. ``None`` on ``evaluate()`` and hand-built results.
        effective_entity_labels: Resolved detection labels used as coverage-prompt
            guidance, without restricting permissive evaluation.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    data_summary: str | None = None
    excluded_entity_labels: list[str] | None = None
    id_column: str | None = None
    detection_scope: DetectionScope | None = None
    effective_entity_labels: list[str] | None = None
    _display_cycle_index: int = field(default=0, init=False, repr=False)
    _outcomes: tuple[TargetOutcome, ...] | None = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            "AnonymizerResult("
            f"rows={len(self.dataframe)}, "
            f"columns={len(self.dataframe.columns)}, "
            f"trace_columns={len(self.trace_dataframe.columns)}, "
            f"failed_records={len(self.failed_records)}"
            ")"
        )


@dataclass
class PreviewResult(_DisplayMixin):
    """Result returned by preview runs.

    Attributes:
        dataframe: User-facing columns only (text, replaced/rewritten text, scores).
        trace_dataframe: Full pipeline trace including all internal columns.
        resolved_text_column: Name of the user-facing text column. Equals the
            user's requested ``text_column`` unless the reader had to rename it
            to avoid colliding with an Anonymizer output column, in which case
            it is the post-rename identifier (e.g. ``"final_entities__input"``).
        failed_records: Records that failed during pipeline processing.
        preview_num_records: Number of records requested for the preview.
        replace_method: The replace strategy that produced this preview. Set by
            ``preview()``; consumed by ``evaluate()`` to dispatch the right
            judges. ``None`` on results that were constructed by hand or loaded
            from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this preview when rewrite
            mode was used. Set by ``preview()``; consumed by ``evaluate()`` to
            dispatch the rewrite judges. Mutually exclusive with ``replace_method``.
        entity_labels: Allowlist of entity labels that were in scope during
            detection. Preserved for ``evaluate()`` so the coverage judge scopes
            its evaluation to the same label set. ``None`` means detection was
            permissive: default labels were configured, and the augmenter may
            have emitted additional labels. Configured examples are intentionally
            not persisted for evaluation.
        data_summary: Optional dataset context supplied with the original input.
            Preserved for ``evaluate()`` so entity-coverage judging uses the
            same context as detection.
        excluded_entity_labels: Labels that were explicitly excluded from
            detection. Preserved for ``evaluate()`` so the coverage judge does
            not penalise the output for not anonymizing excluded labels.
        id_column: Record identifier column retained in ``dataframe`` for
            in-memory record input. File-backed input currently leaves this as
            ``None``.
        detection_scope: The resolved detection scope the run used: effective
            labels, allowed labels (``None`` when ``entity_labels`` is unset),
            exclusions, and GLiNER threshold. ``None`` on ``evaluate()`` and hand-built results.
        effective_entity_labels: Resolved detection labels used as coverage-prompt
            guidance, without restricting permissive evaluation.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    preview_num_records: int
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    data_summary: str | None = None
    excluded_entity_labels: list[str] | None = None
    id_column: str | None = None
    detection_scope: DetectionScope | None = None
    effective_entity_labels: list[str] | None = None
    _display_cycle_index: int = field(default=0, init=False, repr=False)
    _outcomes: tuple[TargetOutcome, ...] | None = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            "PreviewResult("
            f"rows={len(self.dataframe)}, "
            f"columns={len(self.dataframe.columns)}, "
            f"trace_columns={len(self.trace_dataframe.columns)}, "
            f"failed_records={len(self.failed_records)}, "
            f"preview_num_records={self.preview_num_records}"
            ")"
        )
