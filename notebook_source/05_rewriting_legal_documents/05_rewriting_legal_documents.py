# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rewriting Legal Documents
#
# Rewriting legal text (TAB dataset) with a domain-specific privacy goal
# and custom entity labels tailored for legal proceedings.

# %% [markdown]
# ## Setup

# %%
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Detect, LoggingConfig, Rewrite, configure_logging
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal, RiskTolerance

configure_logging(LoggingConfig.debug())

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## Input data
#
# TAB (Text Anonymization Benchmark) legal documents -- court decisions
# containing names, dates, case numbers, and other legal identifiers.

# %%
LEGAL_ENTITY_LABELS = [
    "first_name",
    "last_name",
    "court_name",
    "organization_name",
    "company_name",
    "prison_detention_facility",
    "street_address",
    "city",
    "state",
    "country",
    "date",
    "date_time",
    "time",
    "date_of_birth",
    "age",
    "email",
    "phone_number",
    "ssn",
    "unique_id",
    "legal_role",
    "case_number",
    "application_number",
    "monetary_amount",
    "sentence_duration",
    "nationality",
]

input_data = AnonymizerInput(
    source="../data/TAB_legal_sample25.csv",
    text_column="text",
    data_summary="Legal court decisions containing personal identifiers, case numbers, and institutional references",
)

# %% [markdown]
# ## Rewrite with legal-specific config

# %%
config = AnonymizerConfig(
    detect=Detect(
        entity_labels=LEGAL_ENTITY_LABELS,
    ),
    rewrite=Rewrite(
        privacy_goal=PrivacyGoal(
            protect="All personal identifiers, case numbers, court names, and institutional references that could identify parties",
            preserve="Legal reasoning, procedural facts, statutory references, and the structure of the ruling",
        ),
        evaluation=EvaluationCriteria(
            risk_tolerance=RiskTolerance.strict,
            max_repair_iterations=3,
        ),
    ),
)

preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

preview.display_record(0)

# %%
preview.display_record(1)

# %% [markdown]
# ## Inspect the results

# %%
result = anonymizer.run(config=config, data=input_data)

print(result)
result.dataframe.head()

# %%
result.dataframe[["text_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

# %% [markdown]
# ## Filter by review flag

# %%
df = result.dataframe
flagged = df[df["needs_human_review"] == True]  # noqa: E712
print(f"{len(flagged)} of {len(df)} records flagged for human review")
flagged.head()
