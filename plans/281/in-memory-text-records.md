<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# In-Memory Text Records

Tracks [GitHub issue #281](https://github.com/NVIDIA-NeMo/Anonymizer/issues/281).

## Goal

Let Python callers anonymize text already held in memory without first writing a
temporary CSV or Parquet file. Callers can attach stable record IDs so completed
and failed rows remain correlated with application-owned events.

## Design

Add two explicit public models:

- `TextRecord(id, text)` describes one caller-owned record.
- `TextRecordsInput(records, data_summary)` describes an ordered in-memory batch.

`Anonymizer.run()` accepts `TextRecordsInput` alongside the existing file-backed
`AnonymizerInput`. The reader converts both forms into the same internal
dataframe contract before detection begins. Caller IDs populate Anonymizer's
private record-ID column and are returned with successful and failed records.

The existing `AnonymizerInput.source` field remains required. The CLI continues
to accept file input only, avoiding nested command-line record syntax.

## Compatibility and Safety

This is an additive Python API. Existing file input, schemas, CLI flags, and
configured file-backed ID columns are unchanged. Input order is preserved, and
generated output columns cannot replace the caller's record identity.

## Validation

- unit tests for model validation, ordering, IDs, column collisions, and failures;
- schema regression test proving `AnonymizerInput.source` is still required;
- measurement tests for the in-memory source metadata; and
- the complete repository test suite.

## Non-Goals

- accepting arbitrary dataframes;
- exposing in-memory records through the CLI; or
- changing file-backed ID-column behavior.
