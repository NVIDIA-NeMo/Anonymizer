# Architecture

PLACEHOLDER: 2-3 sentences describing what Anonymizer does, where NDD fits, and what DataDesigner provides.

## Initialization

```mermaid
graph LR
    Anonymizer___init__["Anonymizer.__init__"] --> _initialize_logging["_initialize_logging"]
    Anonymizer___init__["Anonymizer.__init__"] --> Path["Path"]
    Anonymizer___init__["Anonymizer.__init__"] --> parse_model_configs["parse_model_configs"]
    Anonymizer___init__["Anonymizer.__init__"] --> info["info"]
    Anonymizer___init__["Anonymizer.__init__"] --> len["len"]
    Anonymizer___init__["Anonymizer.__init__"] --> DataDesigner["DataDesigner"]
    Anonymizer___init__["Anonymizer.__init__"] --> _resolve_model_providers["_resolve_model_providers"]
    Anonymizer___init__["Anonymizer.__init__"] --> NddAdapter["NddAdapter"]
    Anonymizer___init__["Anonymizer.__init__"] --> EntityDetectionWorkflow["EntityDetectionWorkflow"]
    Anonymizer___init__["Anonymizer.__init__"] --> ReplacementWorkflow["ReplacementWorkflow"]
    Anonymizer___init__["Anonymizer.__init__"] --> LlmReplaceWorkflow["LlmReplaceWorkflow"]
    _initialize_logging["_initialize_logging"] --> configure_logging["configure_logging"]
    parse_model_configs["parse_model_configs"] --> _load_yaml_dict["_load_yaml_dict"]
    parse_model_configs["parse_model_configs"] --> ParsedModelConfigs["ParsedModelConfigs"]
    parse_model_configs["parse_model_configs"] --> load_model_configs["load_model_configs"]
    parse_model_configs["parse_model_configs"] --> load_default_model_selection["load_default_model_selection"]
    parse_model_configs["parse_model_configs"] --> isinstance["isinstance"]
    parse_model_configs["parse_model_configs"] --> _parse_yaml_string["_parse_yaml_string"]
    parse_model_configs["parse_model_configs"] --> pop["pop"]
    parse_model_configs["parse_model_configs"] --> _merge_selections["_merge_selections"]
    _resolve_model_providers["_resolve_model_providers"] --> isinstance["isinstance"]
    _resolve_model_providers["_resolve_model_providers"] --> load_config_file["load_config_file"]
    _resolve_model_providers["_resolve_model_providers"] --> get["get"]
    _resolve_model_providers["_resolve_model_providers"] --> ValueError["ValueError"]
    _resolve_model_providers["_resolve_model_providers"] --> model_validate["model_validate"]
```

## Data Flow

```mermaid
graph LR
    Anonymizer__run_internal["Anonymizer._run_internal"] --> read_input["read_input"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> len["len"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> info["info"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> isEnabledFor["isEnabledFor"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> astype["astype"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> debug["debug"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> perf_counter["perf_counter"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> run["run"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> _count_entities["_count_entities"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> _count_labels["_count_labels"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> _rename_output_columns["_rename_output_columns"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> AnonymizerResult["AnonymizerResult"]
    Anonymizer__run_internal["Anonymizer._run_internal"] --> _build_user_dataframe["_build_user_dataframe"]
    read_input["read_input"] --> _load_dataframe["_load_dataframe"]
    read_input["read_input"] --> InvalidInputError["InvalidInputError"]
    read_input["read_input"] --> _validate_internal_column_collision["_validate_internal_column_collision"]
    read_input["read_input"] --> rename["rename"]
    _count_entities["_count_entities"] --> sum["sum"]
    _count_entities["_count_entities"] --> apply["apply"]
    _count_entities["_count_entities"] --> _unwrap_entities["_unwrap_entities"]
    _count_labels["_count_labels"] --> Counter["Counter"]
    _count_labels["_count_labels"] --> _count_labels_for_row["_count_labels_for_row"]
    _rename_output_columns["_rename_output_columns"] --> rename["rename"]
    _build_user_dataframe["_build_user_dataframe"] --> copy["copy"]
```

## Entrypoints

PLACEHOLDER: Description of public API entry points.

## CI and Developer Workflow

PLACEHOLDER: Description of the CI pipeline and developer workflow.
