# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.anonymizer_config import Detect as _DetectConfig
from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.rewrite.chunked_steps import WindowedStepParams, make_windowed_metadata_generator
from anonymizer.engine.schemas import Domain, DomainClassificationSchema

_DEFAULT_MAX_RENDER_CHARS: int = _DetectConfig.model_fields["detection_window_max_render_chars"].default
_DEFAULT_SAFETY_MARGIN_CHARS: int = _DetectConfig.model_fields["detection_window_safety_margin_chars"].default


def _first_output(outputs: list[Any]) -> dict[str, Any]:
    """Domain is a single doc-level label: keep the first window's classification."""
    return outputs[0].model_dump(mode="json")

# ---------------------------------------------------------------------------
# Single source of truth for rewrite-domain metadata.
#
# Each Domain has one entry in DOMAIN_METADATA carrying:
#   - classification_description: short blurb the classifier prompt shows the
#     LLM so it can pick the right domain.
#   - quality_supplement:         guidance injected via COL_DOMAIN_SUPPLEMENT
#                                 into the meaning-unit extraction prompt.
#   - privacy_supplement:         guidance injected via COL_DOMAIN_SUPPLEMENT_PRIVACY
#                                 into the sensitivity disposition prompt.
#                                 Defaults to ``quality_supplement`` when not
#                                 set explicitly, since most domains use the
#                                 same guidance for both.
#
# Adding a new Domain value without a matching entry here raises at module
# import time (see _build_domain_index below), so drift is caught immediately.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainMetadata:
    """All rewrite-pipeline metadata associated with a single Domain.

    ``privacy_supplement`` defaults to ``quality_supplement`` when not given,
    so domains that share identical privacy and quality guidance can omit it.
    """

    domain: Domain
    classification_description: str
    quality_supplement: str
    privacy_supplement: str | None = None

    def __post_init__(self) -> None:
        if self.privacy_supplement is None:
            object.__setattr__(self, "privacy_supplement", self.quality_supplement)


DOMAIN_METADATA: tuple[DomainMetadata, ...] = (
    DomainMetadata(
        domain=Domain.BIOGRAPHY_PROFILE,
        classification_description="Personal profiles, CVs/resumes, biographical narratives, employee bios",
        quality_supplement="Focus on: core life roles and occupations; long-term activities and commitments; career trajectory and development (including training, education, major transitions, and advancement into current roles); distinctive skills or ways of doing things in those roles (e.g., creative methods, sourcing philosophy, technical or artistic approach); central motivations and formative influences rooted in early experience; and key, ongoing relationships or family structures that shape the individual's life or work.\n\nYou MUST capture high-level educational background and professional trajectory when present, expressed in abstract terms (e.g., advanced study, early-stage training, work at major observatory, move into leadership), even if specific institutions or dates must be generalized.\n\nAlso capture signature outputs or recurring creations that represent the individual's identity or history (e.g., a recurring research theme, a major discovery focus, a signature dish), especially when tied to motivation or heritage.\n\nDrop: street-level or hyper-local locations, exact ages, precise institutions, and identifying anecdotes that do not materially affect development, output, values, or long-term identity.",
    ),
    DomainMetadata(
        domain=Domain.INSURANCE,
        classification_description="Claims, policies, underwriting notes, adjustment reports, benefits correspondence",
        quality_supplement="Capture any information relevant to insurance decision-making, claim adjudication, coverage assessment, or policyholder outcomes. This includes (when present):\n- The type and scope of insurance product (health, auto, property, life, disability, liability, group benefits),\n- The triggering event being claimed or assessed (accident, illness, loss, property damage), described abstractly,\n- Coverage decisions and the documented reasoning (approval, denial, partial payment, settlement, escalation, appeal),\n- Underwriting or risk-assessment factors at a category level (e.g., 'an older adult applicant', 'a high-risk class'),\n- Communication channels or interactions with the insurer (claim filing, adjuster contact, appeals correspondence), abstracted,\n- Evidence of a policy or claim relationship described generically, without retaining policy or claim identifiers,\n- Generalized demographic, residency, or occupational context only when relevant to suitability, risk, or coverage eligibility,\n- High-level monetary outcomes such as total claimed, total paid, or settlement category (generalized when needed).\n\nFor claim records, adjustment reports, or benefits correspondence, ALWAYS capture:\n1) That a claim, adjustment, or benefit decision exists,\n2) What category of loss or service it covers,\n3) The disposition (paid, denied, in review, settled) and the stated reasoning,\n4) The presence of a reporting or coverage period (generalized if needed).\n\nWhen policy numbers, claim numbers, member IDs, group numbers, or other insurer-specific identifiers appear, treat them only as evidence of policy or claim linkage — DO NOT retain any exact identifier values.\n\nMedical, behavioral, or other sensitive details inside health-, disability-, or liability-claim records must be preserved only at category level (e.g., 'an orthopedic injury', 'a chronic condition') and only when central to the adjudication logic; do NOT reproduce specific named diagnoses, providers, or facilities.\n\nAlways extract:\n1) The insurance scenario (product type, coverage scope, claim trigger),\n2) The decision or recommendation and its stated reasoning,\n3) Communication or escalation channels (abstracted),\n4) Risk or eligibility attributes that materially influence coverage,\n5) The existence of a policy, claim, or benefit relationship,\n6) Any reporting period, coverage limit, or outcome that provides meaningful context.\n\nPrefer abstraction over deletion: preserve high-level insurance meaning even when specific identifiers, monetary figures, or named medical details must be removed.",
    ),
    DomainMetadata(
        domain=Domain.GOVERNMENT_PUBLIC_RECORDS,
        classification_description="FOIA responses, public records, official government reports and correspondence",
        quality_supplement="Capture every information-bearing unit needed to understand the substance, posture, and implications of the official record. Government and public-records text often combines dense procedural detail with policy substance; preserve both in abstract form.\n\nAlways capture, if present:\n1) The category of record (FOIA response, agency report, official correspondence, public notice, oversight filing) and the role of the agency behind it (e.g., 'a regulatory agency', 'a federal department', 'a state office'),\n2) The subject or matter the record concerns (program, investigation, decision, complaint, audit, public-interest topic), described abstractly,\n3) Decisions, findings, or actions taken by officials, plus the stated authority or statutory basis,\n4) Procedural context: how the record was triggered or filed, the response timeline (sequence or duration only, not exact dates), and any pending or completed phases,\n5) Stakeholder impacts: who is affected and how (regulated entities, beneficiaries, the public, specific industries, geographic regions at the regional or national level),\n6) References to other records, prior decisions, or policy lineage that frame the current record,\n7) The presence and category of redactions or withheld material (e.g., 'redacted to protect personally identifying information', 'withheld under deliberative-process privilege'); the existence and rationale for a redaction is meaningful even when the redacted content is not.\n\nRefer to officials, requesters, and named individuals only by role or affiliation (e.g., 'an agency administrator', 'a requester', 'an oversight committee'); do NOT retain personal names or specific titles tied to identifiable individuals.\n\nTreat case or record locators, FOIA tracking numbers, docket identifiers, exact monetary figures, exact dates, and named offices below the agency level as high-risk anchors that should be generalized rather than reproduced verbatim.\n\nPreserve generic institutional references (e.g., 'the agency', 'the department', 'a regional office') as structural terms — these are not identifiers and are needed for meaning.\n\nDrop: incidental references to private individuals appearing in the record (witnesses, complainants, third-party employees) unless their role is central to the substance; granular procedural minutiae that does not materially change the record's posture or outcome.\n\nPrefer abstraction over deletion: preserve what the record decides, what authority it acts under, who is affected, and what posture it leaves the matter in, even when specific identifiers must be generalized.",
    ),
    DomainMetadata(
        domain=Domain.NEWS_PUBLIC_AFFAIRS,
        classification_description="Journalism, news articles, press releases, public affairs reporting",
        quality_supplement="Extract every distinct information-bearing unit from the article. Do NOT summarize or compress multiple claims into one unit. A meaning unit is ONE independent idea: a fact, stance, causal link, prediction, stakeholder impact, trend, or description of who is doing what at a role/group/institution level.\n\nAlways capture, if present:\n1) Policy or event: stances, approvals, bans, reversals, regulatory shifts, key decisions.\n2) Actors as roles or groups: parties, governments, business sectors, lobbies, institutions (never names).\n3) Motivations and reasoning: economic, political, ideological justifications on all sides.\n4) Polarity: who supports, who opposes, and on what grounds.\n5) Evidence and analogies used to support arguments (paraphrased, not quoted verbatim).\n6) Stakeholder impacts: effects on farmers, workers, retailers, investors, consumers, or public services.\n7) System-level consequences: investor confidence, monopoly risk, inflation, growth, governance stability.\n8) Broader trends: membership growth, reform momentum, shifts in public sentiment or party credibility.\n9) Policy lineage: how the current stance differs from previous governments or earlier policy.\n10) Temporal framing: before/after elections, reforms, court rulings, crises, or other milestones.\n\n11) Extract historical/political lineage when present (election wins, shifts from prior ruling party).\n\nSegmentation rules:\n• If a sentence contains multiple independent ideas (e.g., a reason AND a separate consequence), split them.\n• If two sentences express one tightly bound idea that cannot stand alone if split, keep them as one unit.\n• When in doubt about splitting, err on the side of creating MORE, smaller units rather than fewer, larger ones.\n\nDo NOT use personal names or specific PII. Refer to actors only by their roles or affiliations (e.g., 'the party', 'a business-sector recruit', 'small-trader lobbies', 'the previous government').\n\nThere is NO fixed number of units required. Continue extracting meaning units until no substantial claim, stance, cause, effect, or stakeholder-impact statement remains unrepresented.",
    ),
    DomainMetadata(
        domain=Domain.MARKETING_COMMERCIAL,
        classification_description="Advertising copy, product descriptions, product reviews, customer feedback, promotional content",
        quality_supplement="Capture the product or service being promoted, the key features being offered, and the benefits or value claims made to the customer.\n\nAlways extract distinct meaning units for:\n- Core features or capabilities of the product or service,\n- Stated benefits such as convenience, security, flexibility, or ease of use,\n- Access conditions (e.g., availability windows, 24/7 access, multi-channel access),\n- Optional enhancements (e.g., linking cards, authentication options, bonus features),\n- Any security mechanisms or protection features referenced.\n\nIf sensitive identifiers (e.g., account numbers, card numbers, routing/codes, biometric IDs) appear in the text, DO NOT reproduce their values. Instead retain the meaning at an abstract level — e.g., 'an associated payment card exists', 'the account supports direct deposits', 'a biometric credential is available for login'.\n\nDo not collapse multiple features into a single unit. Marketing content is feature-enumerative — so keep features separate unless they are truly inseparable in meaning.",
    ),
    DomainMetadata(
        domain=Domain.TECHNICAL_SOFTWARE_ENGINEERING,
        classification_description="Code, logs, configs, API traces, system outputs, developer docs",
        quality_supplement="Focus on: inputs, outputs, constraints, assumptions, requirements, algorithms, interfaces, failure modes, and key configuration logic. Drop: machine hostnames, internal user handles, and environment-specific paths unless essential to semantics.",
    ),
    DomainMetadata(
        domain=Domain.RESEARCH_SCIENTIFIC,
        classification_description="Research papers, academic studies, clinical trials, lab notes, data analyses, survey data",
        quality_supplement="Focus on: research questions, hypotheses, methods, datasets (at an abstract level), core findings, limitations, and implications. Drop: reviewer comments, individual participant anecdotes, and unnecessary bibliographic minutiae.",
    ),
    DomainMetadata(
        domain=Domain.SECURITY_INFOSEC,
        classification_description="Incident reports, vulnerability disclosures, threat models, security logs, pen test reports",
        quality_supplement="Focus on: threat models, vulnerabilities, attack vectors, mitigations, and incident flows. Drop: live credentials, specific machine identifiers, and any data that could enable real-world compromise.",
    ),
    DomainMetadata(
        domain=Domain.FINANCIAL,
        classification_description="Financial statements, bank records, loan docs, investment records, account statements, macro/economic analysis",
        quality_supplement="Capture any information relevant to financial decision-making, risk assessment, customer needs, or product evaluation. This includes (when present):\n- The type and scope of financial products (mortgages, loans, credit lines, investment portfolios, account summaries),\n- Customer actions such as exploring rates, repayment plans, contribution levels, or product options,\n- Communication channels or engagement with the institution (abstracted),\n- Evidence of a financial relationship or account ownership (described generically),\n- Portfolio or loan structure (diversification, term options, repayment strategy),\n- Generalized demographic or residency context when relevant to suitability or risk (e.g., 'a young adult borrower', 'a customer in a regional market'),\n- High-level financial totals or outcomes such as total income, total expenses, projected balance, net surplus/deficit, or reported aggregates.\n\nFor account statements, investment reports, or portfolio summaries, ALWAYS capture:\n1) That a statement/report exists or has been generated,\n2) What it covers (e.g., holdings across asset classes),\n3) What categories of metrics it reports (values, acquisition dates, performance metrics),\n4) The presence of a reporting date or period (generalized if needed).\n\nDemographic or geographic attributes should be preserved only in abstract form and only when relevant to financial interpretation. Remove any exact surface forms listed in the sensitive entities block.\n\nWhen accounts, emails, routing numbers, card numbers, URLs, or institution-specific markers appear, treat them only as evidence of a customer relationship or account linkage — DO NOT retain any exact identifier values.\n\nAlways extract:\n1) The financial scenario (product, purpose, or portfolio scope),\n2) Customer actions or evaluation behaviors,\n3) Communication or engagement channels (abstracted),\n4) Attributes that materially influence financial suitability,\n5) The existence of a financial account or relationship,\n6) Any reporting period or financial outcome that provides meaningful context.\n\nPrefer abstraction over deletion: preserve high-level financial meaning even when specifics must be removed.",
    ),
    DomainMetadata(
        domain=Domain.ECONOMIC_ANALYSIS,
        classification_description="Macroeconomic reports, forecasts, economic research",
        quality_supplement="Focus on: macro or sector-level drivers, indicators, models, and causal reasoning. Drop: overly specific personal anecdotes or local identifiers that don't affect the analysis.",
    ),
    DomainMetadata(
        domain=Domain.POLICY_REGULATORY,
        classification_description="Regulations, compliance documents, policy guidance, standards, enforcement procedures",
        quality_supplement="Focus on: obligations, rights, conditions, thresholds, exceptions, and enforcement mechanisms. Keep rule structure clear. Drop: individual case anecdotes unless they are essential exemplars.",
    ),
    DomainMetadata(
        domain=Domain.LEGAL,
        classification_description="Contracts, court records, legal memos, case files, legal correspondence, enforcement actions",
        quality_supplement="Focus on: (1) the core procedural arc of the case; (2) the legal basis for the application; (3) key domestic decisions and the reasons given; (4) treatment of detention, remand orders, extensions, and risks cited; (5) indictment, charges, and major trial-stage actions; (6) essential appeal outcomes; (7) any rights-interference events (e.g., censorship, lack of counsel). Always capture: (a) notification to the respondent Government and identification of Government representation; (b) the basic hearing chronology (at minimum the sequence or number of hearings, even if dates are abstracted); (c) any concurrent-sentence or sentence-merging orders; (d) the decision to join merits with admissibility or other major procedural consolidations. Drop: specific dates, place names, identities of individuals, and granular factual background unless they materially change the legal reasoning or procedural posture.\n\nWhen the text is a legal form, authorization document, contract template, or procedural instrument, also treat the following as meaning-bearing elements that must be extracted in abstract form: \n- The presence of a 'customer information' or 'party information' section, captured abstractly (e.g., the form includes fields for the individual's identifying and account-related information). \n- The presence of financial or account-related fields needed to execute the authorization (e.g., payment instruments, account relationships, routing elements), expressed generically without repeating sensitive identifiers. \n- The structural components of the document such as required fields, placeholders, or sections (e.g., signature block, date field, merchant details, amount fields), when they define the meaning or operation of the document. \n- The fact that the form is a template requiring user-provided inputs (e.g., amount, date, merchant details), expressed abstractly. \n\nDemographic quasi-identifiers (e.g., age, education level, nationality, ethnicity, gender) must be DROPPED unless they are CENTRAL to the legal reasoning, eligibility determination, statutory thresholds, contractual validity, or procedural rights. If such attributes appear but do not affect the legal meaning or enforceability of the document, they must NOT be turned into meaning units and must only appear, if at all, in highly abstract phrasing (e.g., 'general background information'), without preserving the specific attribute.",
        privacy_supplement='Apply the following domain-specific privacy guidance for LEGAL text.\n\n- Treat exact court names, tribunal names, exact monetary amounts, adjudicating bodies, prison or detention-facility names, unique case or record locators, and exact sentencing details as high-risk legal anchors that MUST be protected.\n- ALWAYS protect names of lawyers and other legal professionals when they refer to specific individuals.\n- Use generalized forms such as "a trial court", "an appellate court", "a national court", "a detention facility", or "a prison" instead of specific institutional names.\n- Do NOT treat generic institutional references (e.g., "the court", "a regional court", "an appellate court") as identifiers. These are structural terms and MUST be preserved.\n- In legal text, exact place names, exact dates, charge details, procedural milestones, and detailed counts may become identifying in combination and should be generalized when they materially increase person-, case-, or record-linkage risk.\n- Rare combinations of allegations, procedural history, court progression, sentence structure, appeal outcomes, remedies sought, and institutional context may function as latent identifiers; break these bundles by generalizing or suppressing one or more supporting details.\n- Preserve abstract legal meaning, party roles, and high-level procedural chronology where possible without retaining exact identifying detail.\n- Preserve generic professional roles such as "lawyer", "judge", or "prosecutor" ONLY when used as role labels and not tied to identifiable individuals, law firms, or specific cases.\n',
    ),
    DomainMetadata(
        domain=Domain.HR_EMPLOYMENT,
        classification_description="Employee records, performance reviews, job applications, compensation data, HR correspondence",
        quality_supplement="Focus on: roles, responsibilities, performance dimensions, behavioral expectations, and process structure (e.g., review cycles). Drop: personal gossip, unnecessary names, and exact dates of incidents unless structurally important.",
    ),
    DomainMetadata(
        domain=Domain.BUSINESS_OPERATIONS,
        classification_description="Strategy docs, project plans, meeting notes, internal memos, operational materials",
        quality_supplement="Focus on: goals, KPIs, processes, decision criteria, resource allocation, and coordination patterns. Drop: individual identities and low-level scheduling minutiae unless essential.",
    ),
    DomainMetadata(
        domain=Domain.MEDICAL_CLINICAL,
        classification_description="Clinical notes, EHR, patient records, health forms, medical correspondence, medical transcripts",
        quality_supplement="Focus on: symptoms, diagnoses, assessments, clinical reasoning, interventions, timelines of care, and outcomes. The goal is to preserve the clinical story and decision-making logic while removing or abstracting details that could contribute to re-identification.\n\nAlways keep, in appropriately abstract form:\n- Presenting symptoms, key past history elements that drive current assessment, and relevant exam findings,\n- The main working diagnoses or differential diagnoses (generalized when needed),\n- The clinician's reasoning about cause, risk, and prognosis,\n- Interventions, medications, and monitoring plans (prefer generic drug classes or roles over brand names unless the specific agent is critical to the clinical meaning),\n- High-level timelines of care (e.g., 'during childhood', 'over several years', 'at follow-up') rather than exact dates or ages.\n\nClarification and education requests:\n- When a patient explicitly asks for explanations or definitions of medical, anatomical, imaging, or diagnostic terms (e.g., nerve root abutment, annular bulge or tear, MRI findings), you MUST preserve those terms in generalized clinical language rather than collapsing them into vague references.\n- Treat these as diagnostic-understanding or clinical-education intents, not as symptom reports. Capture both what concept the patient is asking about and how the clinician responds (explains, defers, or refers).\n\nNamed conditions and findings:\n- Preserve non-identifying anatomical and pathological terms (e.g., nerve root contact, disc bulge, annular tear) when they are central to the patient's question or the clinician's reasoning.\n- Do NOT drop or over-generalize such terms if doing so would remove the substantive clinical concept or the educational value of the exchange.\n\nShort clinical exchanges:\n- In brief patient–clinician interactions (such as single-question consultations), prioritize capturing:\n  • The specific medical concepts or findings being asked about,\n  • The nature of the clinician's response (explanation, reassurance, deferral, or referral),\n  even if no formal diagnosis or complete treatment plan is provided.\n\nBenign viral conditions:\n- When benign, self-limited viral illnesses are mentioned (e.g., 'viral upper respiratory infection', 'common cold', 'flu-like viral illness'), you MUST generalize them to a high-level phrase such as 'a common viral illness' or 'a routine viral respiratory infection'.\n- Do NOT repeat the exact surface form from the sensitive entities block for these conditions, unless the specific named pathogen or syndrome is central to the clinical reasoning (e.g., a disease-specific treatment or public-health implication). In most primary-care-style counseling notes, abstraction to 'a common viral illness' is preferred.\n\nHandle sensitive attributes carefully:\n- Demographics (age, gender, ethnicity, detailed location) should be represented only in coarse categories (e.g., 'a young adult', 'an older adult') and only when they materially influence risk, diagnosis, or management.\n- Family medical history and genetic predispositions should be preserved only at the level needed for clinical logic (e.g., 'a family history of a related endocrine disorder') without copying exact surface forms.\n- Stigmatizing or highly sensitive conditions (e.g., certain mental health diagnoses, sexually transmitted infections, substance use) should be abstracted to category-level descriptions unless the specific label is necessary to understand the clinical decision-making.\n\nAbstract or drop:\n- Highly identifying combinations of demographics, rare conditions, and free-text anecdotes that are not essential to the clinical interpretation,\n- Exact dates, precise ages, and detailed timelines when a higher-level temporal description suffices,\n- Institution names, clinician names, and operational details that do not affect assessment or plan.\n\nWhen in doubt, preserve the clinical reasoning, the key medical concepts, and the care trajectory in generalized language, and prefer abstraction over deletion so that the medical meaning remains intact while sensitive surface forms are removed.",
    ),
    DomainMetadata(
        domain=Domain.EDUCATION,
        classification_description="Student records, educational assessments, academic transcripts, instructional correspondence",
        quality_supplement="Focus on: learning objectives, knowledge structure, exercises, and evaluation criteria. Drop: specific student identifiers or one-off comments that do not change the instructional content.",
    ),
    DomainMetadata(
        domain=Domain.CREATIVE_FICTION,
        classification_description="Fiction, scripts, creative writing with embedded real-world names or places",
        quality_supplement="Focus on: key plot events, character arcs, world rules, and central themes. Drop: extraneous descriptive flourishes and minor side characters that do not affect story coherence.",
    ),
    DomainMetadata(
        domain=Domain.ENTERTAINMENT_MEDIA,
        classification_description="Non-fiction media content, shows, podcasts, celebrity coverage",
        quality_supplement="Focus on extracting distinct information-bearing units about media content, formats, celebrity behavior, and cultural themes. Do NOT summarize or compress multiple ideas into one unit if they can be separated.\n\nTreat as separate meaning units when possible:\n• Each show, segment, or series mentioned (news specials, talk shows, documentaries, reality shows).\n• Each description of format or style (dramatic reconstructions, panel debate, visual style, tone).\n• Each opinion or evaluative judgment about a show, season, host, or guest.\n• Each notable celebrity appearance or interaction (who appears with whom, in what context, doing what).\n• Each cultural or social theme raised (gender roles, sexuality, family dynamics, fame, audience tastes).\n• Each programming or scheduling decision (special coverage, repeats tied to new releases, new feature launches).\n• Each observed trend in channel or industry behavior (early election coverage, increasingly suggestive content, etc.).\n\nSegmentation bias: when a sentence contains multiple shows, guests, format choices, or opinions, prefer splitting into multiple meaning units rather than merging them, as long as each unit still conveys a coherent, self-contained claim.",
    ),
    DomainMetadata(
        domain=Domain.SOCIAL_COMMENTARY,
        classification_description="Opinion pieces, editorials, social media posts, cultural criticism, personal blogs",
        quality_supplement="Focus on: main argument, supporting points, evidence, and counterpoints. Drop: gratuitous personal attacks, identifying anecdotes about third parties, and unnecessary specifics that don't change the argument.",
    ),
    DomainMetadata(
        domain=Domain.META_TEXT,
        classification_description="Dataset documentation, annotation guidelines, labeling instructions, prompt templates",
        quality_supplement="Focus on: labels, annotation schemas, instructions, and configuration semantics. Drop: incidental examples that leak real-world identifiers if they are not central to understanding the schema.",
    ),
    DomainMetadata(
        domain=Domain.OTHER,
        classification_description="Content that clearly doesn't fit any of the above domains.",
        quality_supplement="When the text doesn't clearly fit any domain, use general judgment: keep the core purpose, main actions, decisions, and outcomes; drop hyper-specific identifiers.",
    ),
)


def _build_domain_index(metadata: tuple[DomainMetadata, ...]) -> dict[Domain, DomainMetadata]:
    """Build the ``Domain -> DomainMetadata`` index, failing on duplicates or
    missing enum coverage.

    Folding both checks into the index build keeps the module free of a
    side-effecting validator that callers must remember to invoke.
    """
    index: dict[Domain, DomainMetadata] = {}
    for entry in metadata:
        if entry.domain in index:
            raise RuntimeError(f"DOMAIN_METADATA contains duplicate entry for Domain.{entry.domain.name}")
        index[entry.domain] = entry
    missing = set(Domain) - index.keys()
    if missing:
        raise RuntimeError(f"DOMAIN_METADATA missing entries for Domain values: {sorted(d.name for d in missing)}")
    return index


_DOMAIN_BY_ENUM: dict[Domain, DomainMetadata] = _build_domain_index(DOMAIN_METADATA)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _get_domain_classification_prompt(data_summary: str | None = None) -> str:
    domain_descriptions = [f'- "{meta.domain.value}":\n  {meta.classification_description}' for meta in DOMAIN_METADATA]
    domains_text = "\n\n".join(domain_descriptions)

    data_context_section = ""
    if data_summary and data_summary.strip():
        # TODO: align entity detection prompts (validation, augment, latent) to use "Dataset description:" label
        data_context_section = (
            f"\n<data_context>\nDataset description: {data_summary.strip()}\n\n"
            "Use this context to help disambiguate when the text could fit multiple domains.\n"
            "</data_context>\n"
        )

    prompt = """Classify the text into ONE primary domain from the available options.

Choose the domain that best reflects how the text is meant to be used or interpreted overall
(its main function), not just any topic it happens to mention.
<<DATA_CONTEXT>>
<domains>
<<DOMAINS>>
</domains>

<disambiguation_guidelines>
- If the text is a narrative about a worker's life at a company → "BIOGRAPHY_PROFILE", not "BUSINESS_OPERATIONS".
- If the text matches two domains, choose the dominant purpose and structure.
</disambiguation_guidelines>

<text_to_classify>
<<TEXT>>
</text_to_classify>"""
    return substitute_placeholders(
        prompt,
        {
            "<<TEXT>>": _jinja(COL_TEXT),
            "<<DOMAINS>>": domains_text,
            "<<DATA_CONTEXT>>": data_context_section,
        },
    )


# ---------------------------------------------------------------------------
# Custom column generators
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_DOMAIN])
def _enrich_domain(row: dict[str, Any]) -> dict[str, Any]:
    """Attach domain-specific rewrite guidance."""
    parsed_domain = DomainClassificationSchema.model_validate(row[COL_DOMAIN])
    row[COL_DOMAIN_SUPPLEMENT] = _DOMAIN_BY_ENUM[parsed_domain.domain].quality_supplement
    return row


@custom_column_generator(required_columns=[COL_DOMAIN])
def _enrich_domain_privacy(row: dict[str, Any]) -> dict[str, Any]:
    """Attach domain-specific privacy guidance."""
    parsed_domain = DomainClassificationSchema.model_validate(row[COL_DOMAIN])
    row[COL_DOMAIN_SUPPLEMENT_PRIVACY] = _DOMAIN_BY_ENUM[parsed_domain.domain].privacy_supplement
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DomainClassificationWorkflow:
    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        data_summary: str | None = None,
    ) -> list[ColumnConfigT]:
        domain_alias = resolve_model_alias("domain_classifier", selected_models)
        return [
            CustomColumnConfig(
                name=COL_DOMAIN,
                generator_function=make_windowed_metadata_generator(
                    alias=domain_alias,
                    required_columns=[COL_TEXT],
                    schema=DomainClassificationSchema,
                    merge_fn=_first_output,
                    purpose_prefix="domain-classification",
                ),
                generator_params=WindowedStepParams(
                    alias=domain_alias,
                    prompt_template=_get_domain_classification_prompt(data_summary),
                    output_column=COL_DOMAIN,
                    text_column=COL_TEXT,
                    max_render_chars=_DEFAULT_MAX_RENDER_CHARS,
                    safety_margin_chars=_DEFAULT_SAFETY_MARGIN_CHARS,
                    first_only=True,
                ),
            ),
            CustomColumnConfig(
                name=COL_DOMAIN_SUPPLEMENT,
                generator_function=_enrich_domain,
            ),
            CustomColumnConfig(
                name=COL_DOMAIN_SUPPLEMENT_PRIVACY,
                generator_function=_enrich_domain_privacy,
            ),
        ]
