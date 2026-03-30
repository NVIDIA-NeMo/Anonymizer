from pathlib import Path

try:
    NOTEBOOK_SOURCE_DIR = Path(__file__).resolve().parent
except NameError:
    NOTEBOOK_SOURCE_DIR = Path.cwd().parent / "notebook_source"

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, LoggingConfig, Rewrite, configure_logging

configure_logging(LoggingConfig.debug())


anonymizer = Anonymizer()
input_data = AnonymizerInput(
    source=str(NOTEBOOK_SOURCE_DIR / "data" / "synth_bios_sample10.csv"),
    text_column="bio",
    data_summary="Biographical profiles",
)
config = AnonymizerConfig(rewrite=Rewrite())

preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

result = anonymizer.run(config=config, data=input_data)

print(result)
result.dataframe.head()


result.dataframe[["bio_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

print(result.trace_dataframe.columns.tolist())
