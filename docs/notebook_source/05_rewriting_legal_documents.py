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
# # 🕵️ Rewriting Legal Documents
#
# Rewriting legal text (TAB dataset) with a domain-specific privacy goal
# and custom entity labels tailored for legal proceedings.
#
# #### 📚 What you'll learn
#
# - Define domain-specific entity labels for legal text (case numbers, court names, etc.)
# - Configure rewrite mode with legal-specific privacy goals
# - Preview and run on court decision documents
# - Triage flagged records with `needs_human_review`
#
# > **Tip:** First time running notebooks? Start with
# > [setup instructions](https://nvidia-nemo.github.io/Anonymizer/latest/tutorials/).

# %% [markdown]
# ## ⚙️ Setup
#
# - Check if your `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) is registered for model access.
# - Import `Detect` (for custom entity labels), `Rewrite`, and its config classes.
# - `Anonymizer()` initializes with the default model provider -- no extra config needed.
# - `Anonymizer.configure_logging()` controls verbosity -- switch to `Anonymizer.configure_logging(LoggingConfig.debug())` when troubleshooting.


# %%
import getpass
import os

if not os.getenv("NVIDIA_API_KEY"):
    key = getpass.getpass("Enter NVIDIA_API_KEY from build.nvidia.com: ").strip()
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is required to run these notebooks.")
    os.environ["NVIDIA_API_KEY"] = key

# %%
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Detect, Rewrite, configure_logging
from anonymizer.config.rewrite import PrivacyGoal

configure_logging(enabled=False)

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## 📦 Input data
#
# - [TAB (Text Anonymization Benchmark)](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
#   legal documents -- court decisions containing names, dates, case numbers, and other legal identifiers.
# - `LEGAL_ENTITY_LABELS` defines the domain-specific entity types to detect.
#   This replaces the default label set with one tailored to legal text.

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
# ## 🎛️ Configure
#
# - `Detect(entity_labels=...)` overrides the default entity set with legal-specific labels.
# - `PrivacyGoal` tells the rewriter what to **protect** (identifiers, case numbers,
#   institutional references) and what to **preserve** (legal reasoning, statutory references,
#   ruling structure).

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
        risk_tolerance="minimal",
        max_repair_iterations=3,
    ),
)

# %% [markdown]
# ## 👁️ Preview
#
# - Preview on a few records to check that legal entities are detected
#   and the rewrite preserves the ruling's structure.

# %%
preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

preview.display_record(0)

# %%
preview.display_record(1)

# %% [markdown]
# ## 🚀 Full run
#
# - `result.dataframe` has user-facing columns: rewritten text, scores, and the review flag.

# %%
result = anonymizer.run(config=config, data=input_data)

result.dataframe.head()

# %%
result.dataframe[["text_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

# %% [markdown]
# ## 🚩 Filter by review flag
#
# - Records where automated metrics exceed thresholds are flagged for manual review.
# - Use this to prioritize human attention on the records that need it most.
# - See [Working with flagged records](../concepts/rewrite.md#working-with-flagged-records)
#   for guidance on diagnosing and resolving flagged records.

# %%
df = result.dataframe
flagged = df[df["needs_human_review"] == True]  # noqa: E712
print(f"{len(flagged)} of {len(df)} records flagged for human review")
flagged.head()

# %% [markdown]
# ## ⏭️ Next steps
#
# - **[🔍 Inspecting Detected Entities](02_inspecting_detected_entities.ipynb)** --
#   debug what the detection pipeline found before rewriting.
# - **Try it on your own data!** Swap in your CSV, define entity labels for your
#   domain, and set a `PrivacyGoal` that fits -- you've got all the building blocks.
