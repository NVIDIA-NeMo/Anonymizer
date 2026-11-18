# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which schema each nested (JSON-shaped) pipeline column holds.

A column mapped to a model writes its cells with ``model_dump(mode="json")`` and reads
them back with ``read_cell(model, raw)``. A column mapped to ``None`` holds plain JSON
(a dict, list, string or bool, such as the label classification dict) with no model.
"""

from __future__ import annotations

from pydantic import BaseModel

from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTATION_ADMISSION_RESULTS,
    COL_AUGMENTATION_ATTEMPTS,
    COL_AUGMENTATION_DETECTION_RESULT,
    COL_AUGMENTED_CANDIDATES,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_DISPOSITION_COVERAGE,
    COL_DISPOSITION_EXPLICIT_ENTITIES,
    COL_DISPOSITION_LATENT_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_ENTITY_CLASSIFICATION,
    COL_ENTITY_CLASSIFICATION_LLM,
    COL_ENTITY_CLASSIFICATION_LOCAL,
    COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_MERGED_ENTITIES,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_RAW_SENSITIVITY_DISPOSITION,
    COL_RECONCILED_CANDIDATES,
    COL_REPLACEMENT_MAP,
    COL_SEED_DETECTION_RESULT,
    COL_SEED_ENTITIES,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_SENSITIVITY_DISPOSITION,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    COL_VALIDATION_SKELETON,
)
from anonymizer.engine.schemas.detection import (
    AdmissionResults,
    AdmittedCandidates,
    AugmentationAttempts,
    AugmentedEntitiesSchema,
    DetectionResult,
    EntitiesByValueSchema,
    EntitiesSchema,
    LatentEntitiesSchema,
    ValidationCandidatesSchema,
    ValidationRequestDecisionSets,
)
from anonymizer.engine.schemas.replace import EntityReplacementMapSchema
from anonymizer.engine.schemas.rewrite import (
    EntityLabelClassificationsSchema,
    PrivacyAnswersSchema,
    PrivacyQAPairsSchema,
    QACompareResultsSchema,
    QualityAnswersSchema,
    QualityQAPairsSchema,
    RawSensitivityDispositionSchema,
    SensitivityDispositionSchema,
)

NESTED_COLUMN_SCHEMAS: dict[str, type[BaseModel] | None] = {
    # Detection
    COL_SEED_DETECTION_RESULT: DetectionResult,
    COL_SEED_ENTITIES: EntitiesSchema,
    COL_SEED_VALIDATION_CANDIDATES: ValidationCandidatesSchema,
    COL_VALIDATION_SKELETON: None,  # prompt dict with fields beyond ValidationSkeletonSchema
    COL_VALIDATION_INVENTORIES: None,  # list of ValidationRequestInventory
    COL_VALIDATION_DECISIONS: ValidationRequestDecisionSets,
    COL_ADMISSION_RESULTS: AdmissionResults,
    COL_ADMITTED_CANDIDATES: AdmittedCandidates,
    COL_AUGMENTED_ENTITIES: AugmentedEntitiesSchema,
    COL_AUGMENTATION_DETECTION_RESULT: DetectionResult,
    COL_AUGMENTATION_ADMISSION_RESULTS: AdmissionResults,
    COL_AUGMENTATION_ATTEMPTS: AugmentationAttempts,
    COL_AUGMENTED_CANDIDATES: AdmittedCandidates,
    COL_RECONCILED_CANDIDATES: AdmittedCandidates,
    COL_MERGED_ENTITIES: EntitiesSchema,
    COL_VALIDATION_CANDIDATES: ValidationCandidatesSchema,
    COL_DETECTED_ENTITIES: EntitiesSchema,
    COL_FINAL_ENTITIES: EntitiesSchema,
    COL_ENTITIES_BY_VALUE: EntitiesByValueSchema,
    COL_LATENT_ENTITIES: LatentEntitiesSchema,
    # Replace
    COL_ENTITIES_FOR_REPLACE: None,  # list of template-ready entity dicts
    COL_REPLACEMENT_MAP: EntityReplacementMapSchema,
    # Rewrite
    COL_ENTITY_CLASSIFICATION: None,  # label -> category dict
    COL_ENTITY_CLASSIFICATION_LOCAL: None,  # label -> category dict
    COL_ENTITY_CLASSIFICATION_LLM: EntityLabelClassificationsSchema,
    COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS: None,  # list of labels
    COL_DISPOSITION_EXPLICIT_ENTITIES: None,  # numbered prompt lines
    COL_DISPOSITION_LATENT_ENTITIES: None,  # numbered prompt lines
    COL_DISPOSITION_COVERAGE: None,  # bool
    COL_RAW_SENSITIVITY_DISPOSITION: RawSensitivityDispositionSchema,
    COL_SENSITIVITY_DISPOSITION: SensitivityDispositionSchema,
    COL_QUALITY_QA: QualityQAPairsSchema,
    COL_QUALITY_QA_REANSWER: QualityAnswersSchema,
    COL_QUALITY_QA_COMPARE: QACompareResultsSchema,
    COL_PRIVACY_QA: PrivacyQAPairsSchema,
    COL_PRIVACY_QA_REANSWER: PrivacyAnswersSchema,
}
