# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import COL_DOMAIN, COL_DOMAIN_SUPPLEMENT, COL_TEXT, _jinja
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.schemas import Domain, DomainClassificationSchema

# ---------------------------------------------------------------------------
# Domain list — short descriptions used in the domain classification prompt
# to help the LLM identify which domain a given input text belongs to.
# ---------------------------------------------------------------------------

_DOMAIN_LIST: list[tuple[Domain, str]] = [
    (Domain.BIOGRAPHY, "Personal or professional life stories, profiles, narrative descriptions of a person."),
    (
        Domain.CHAT_EMAIL_CSAT,
        "Chats, emails, messaging threads, customer support transcripts, internal correspondence.",
    ),
    (Domain.PRODUCT_REVIEW, "Reviews or ratings of products or services, user-experience writeups, app/store reviews."),
    (Domain.NEWS_JOURNALISM, "News articles, reports, journalistic pieces describing events or situations."),
    (
        Domain.MARKETING_ADVERTISING,
        "Ad copy, campaign messaging, landing page text, brand positioning, promotional material.",
    ),
    (
        Domain.TECHNICAL_ENGINEERING_SOFTWARE,
        "Specifications, design docs, API docs, bug reports, system logs, code-related explanation.",
    ),
    (
        Domain.SCIENTIFIC_ACADEMIC,
        "Research papers, abstracts, methods, results sections, literature reviews, academic essays.",
    ),
    (Domain.SECURITY_INFOSEC, "Security advisories, incident reports, threat models, vulnerability descriptions."),
    (Domain.FINANCIAL, "Financial statements, quantitative analysis, portfolio or investment discussion, budgets."),
    (Domain.ECONOMIC, "Macroeconomic or sector analyses, policy impact assessments, economic modeling discussion."),
    (
        Domain.POLICY_REGULATORY_COMPLIANCE,
        "Organizational policies, regulatory guidance, compliance manuals and procedures.",
    ),
    (Domain.LEGAL, "Contracts, terms of service, court decisions, legal memos, statutory analysis."),
    (Domain.HR_PEOPLE_OPS, "Job descriptions, performance reviews, hiring rubrics, HR policies, people-ops docs."),
    (
        Domain.MANAGEMENT_OPERATIONS,
        "Project plans, OKRs, operational playbooks, strategy docs, internal planning material.",
    ),
    (
        Domain.CLINICAL_EHR_MEDICAL,
        "Clinical notes, EHR snippets, discharge summaries, triage notes, medical documentation.",
    ),
    (Domain.EDUCATIONAL_PEDAGOGICAL, "Lesson plans, curricula, teaching materials, study guides, exam instructions."),
    (
        Domain.FICTION_CREATIVE,
        "Stories, novels, poems, scripts, creative worldbuilding and character-driven narratives.",
    ),
    (
        Domain.ENTERTAINMENT_MEDIA,
        "Film/music/book/game reviews, media commentary, fan writeups about entertainment content.",
    ),
    (Domain.SOCIAL_CULTURAL_OPED, "Opinion pieces, social commentary, cultural criticism, editorials."),
    (
        Domain.PROCEDURAL_INSTRUCTIONAL,
        "How-to guides, recipes, standard operating procedures, instructions and checklists.",
    ),
    (Domain.META_TEXT, "Annotation guidelines, labeling instructions, dataset documentation, prompt templates."),
    (Domain.SOCIAL_MEDIA, "Short posts or comment-style content (tweets, microblogs, captions, comment threads)."),
    (Domain.TRANSCRIPTS_INTERVIEWS, "Interview transcripts, meeting transcripts, Q&A sessions, verbatim call logs."),
    (Domain.OTHER, "Use only if the text clearly does not fit any of the above domains."),
]

# ---------------------------------------------------------------------------
# Domain supplement map — injected into the meaning unit extraction and
# sensitivity disposition prompts via COL_DOMAIN_SUPPLEMENT to tell the LLM
# what to preserve vs. drop when processing text in each domain.
# ---------------------------------------------------------------------------

DOMAIN_SUPPLEMENT_MAP: dict[Domain, str] = {
    Domain.BIOGRAPHY: (
        "Focus on: core life roles and occupations; long-term activities and commitments; "
        "career trajectory and development (including training, education, major transitions, "
        "and advancement into current roles); distinctive skills or ways of doing things "
        "in those roles (e.g., creative methods, sourcing philosophy, technical or artistic "
        "approach); central motivations and formative influences rooted in early experience; "
        "and key, ongoing relationships or family structures that shape the individual's life "
        "or work.\n\n"
        "You MUST capture high-level educational background and professional trajectory when "
        "present, expressed in abstract terms (e.g., advanced study, early-stage training, "
        "work at major observatory, move into leadership), even if specific institutions or "
        "dates must be generalized.\n\n"
        "Also capture signature outputs or recurring creations that represent the individual's "
        "identity or history (e.g., a recurring research theme, a major discovery focus, a "
        "signature dish), especially when tied to motivation or heritage.\n\n"
        "Drop: street-level or hyper-local locations, exact ages, precise institutions, and "
        "identifying anecdotes that do not materially affect development, output, values, "
        "or long-term identity."
    ),
    Domain.CHAT_EMAIL_CSAT: (
        "Focus on: what problem is being discussed, key actions taken, decisions made, and "
        "final outcomes or resolutions. Drop: greetings, sign-offs, small talk, exact phone "
        "numbers, ticket IDs, and email signatures unless truly critical for understanding."
    ),
    Domain.PRODUCT_REVIEW: (
        "Focus on: product features, qualities, user experience, clear pros/cons, issues, "
        "and overall evaluation or recommendation. Drop: specific order IDs, store locations, "
        "shipping details, and identifying anecdotes about the reviewer or third parties."
    ),
    Domain.NEWS_JOURNALISM: (
        "Extract every distinct information-bearing unit from the article. Do NOT summarize or compress multiple "
        "claims into one unit. A meaning unit is ONE independent idea: a fact, stance, causal link, prediction, "
        "stakeholder impact, trend, or description of who is doing what at a role/group/institution level.\n\n"
        "Always capture, if present:\n"
        "1) Policy or event: stances, approvals, bans, reversals, regulatory shifts, key decisions.\n"
        "2) Actors as roles or groups: parties, governments, business sectors, lobbies, institutions (never names).\n"
        "3) Motivations and reasoning: economic, political, ideological justifications on all sides.\n"
        "4) Polarity: who supports, who opposes, and on what grounds.\n"
        "5) Evidence and analogies used to support arguments (paraphrased, not quoted verbatim).\n"
        "6) Stakeholder impacts: effects on farmers, workers, retailers, investors, consumers, or public services.\n"
        "7) System-level consequences: investor confidence, monopoly risk, inflation, growth, governance stability.\n"
        "8) Broader trends: membership growth, reform momentum, shifts in public sentiment or party credibility.\n"
        "9) Policy lineage: how the current stance differs from previous governments or earlier policy.\n"
        "10) Temporal framing: before/after elections, reforms, court rulings, crises, or other milestones.\n\n"
        "11) Extract historical/political lineage when present (election wins, shifts from prior ruling party).\n\n"
        "Segmentation rules:\n"
        "• If a sentence contains multiple independent ideas (e.g., a reason AND a separate consequence), split them.\n"
        "• If two sentences express one tightly bound idea that cannot stand alone if split, keep them as one unit.\n"
        "• When in doubt about splitting, err on the side of creating MORE, smaller units rather than fewer, larger ones.\n\n"
        "Do NOT use personal names or specific PII. Refer to actors only by their roles or affiliations "
        "(e.g., 'the party', 'a business-sector recruit', 'small-trader lobbies', 'the previous government').\n\n"
        "There is NO fixed number of units required. Continue extracting meaning units until no substantial claim, "
        "stance, cause, effect, or stakeholder-impact statement remains unrepresented."
    ),
    Domain.MARKETING_ADVERTISING: (
        "Capture the product or service being promoted, the key features being offered, and the "
        "benefits or value claims made to the customer.\n\n"
        "Always extract distinct meaning units for:\n"
        "- Core features or capabilities of the product or service,\n"
        "- Stated benefits such as convenience, security, flexibility, or ease of use,\n"
        "- Access conditions (e.g., availability windows, 24/7 access, multi-channel access),\n"
        "- Optional enhancements (e.g., linking cards, authentication options, bonus features),\n"
        "- Any security mechanisms or protection features referenced.\n\n"
        "If sensitive identifiers (e.g., account numbers, card numbers, routing/codes, biometric IDs) "
        "appear in the text, DO NOT reproduce their values. Instead retain the meaning at an abstract level — "
        "e.g., 'an associated payment card exists', 'the account supports direct deposits', "
        "'a biometric credential is available for login'.\n\n"
        "Do not collapse multiple features into a single unit. Marketing content is feature-enumerative — "
        "so keep features separate unless they are truly inseparable in meaning."
    ),
    Domain.TECHNICAL_ENGINEERING_SOFTWARE: (
        "Focus on: inputs, outputs, constraints, assumptions, requirements, algorithms, "
        "interfaces, failure modes, and key configuration logic. Drop: machine hostnames, "
        "internal user handles, and environment-specific paths unless essential to semantics."
    ),
    Domain.SCIENTIFIC_ACADEMIC: (
        "Focus on: research questions, hypotheses, methods, datasets (at an abstract level), "
        "core findings, limitations, and implications. Drop: reviewer comments, individual "
        "participant anecdotes, and unnecessary bibliographic minutiae."
    ),
    Domain.SECURITY_INFOSEC: (
        "Focus on: threat models, vulnerabilities, attack vectors, mitigations, and incident "
        "flows. Drop: live credentials, specific machine identifiers, and any data that could "
        "enable real-world compromise."
    ),
    Domain.FINANCIAL: (
        "Capture any information relevant to financial decision-making, risk assessment, customer needs, "
        "or product evaluation. This includes (when present):\n"
        "- The type and scope of financial products (mortgages, loans, credit lines, investment portfolios, "
        "account summaries),\n"
        "- Customer actions such as exploring rates, repayment plans, contribution levels, or product options,\n"
        "- Communication channels or engagement with the institution (abstracted),\n"
        "- Evidence of a financial relationship or account ownership (described generically),\n"
        "- Portfolio or loan structure (diversification, term options, repayment strategy),\n"
        "- Generalized demographic or residency context when relevant to suitability or risk (e.g., 'a young "
        "adult borrower', 'a customer in a regional market'),\n"
        "- High-level financial totals or outcomes such as total income, total expenses, projected balance, "
        "net surplus/deficit, or reported aggregates.\n\n"
        "For account statements, investment reports, or portfolio summaries, ALWAYS capture:\n"
        "1) That a statement/report exists or has been generated,\n"
        "2) What it covers (e.g., holdings across asset classes),\n"
        "3) What categories of metrics it reports (values, acquisition dates, performance metrics),\n"
        "4) The presence of a reporting date or period (generalized if needed).\n\n"
        "Demographic or geographic attributes should be preserved only in abstract form and only when "
        "relevant to financial interpretation. Remove any exact surface forms listed in the sensitive "
        "entities block.\n\n"
        "When accounts, emails, routing numbers, card numbers, URLs, or institution-specific markers appear, "
        "treat them only as evidence of a customer relationship or account linkage — DO NOT retain any exact "
        "identifier values.\n\n"
        "Always extract:\n"
        "1) The financial scenario (product, purpose, or portfolio scope),\n"
        "2) Customer actions or evaluation behaviors,\n"
        "3) Communication or engagement channels (abstracted),\n"
        "4) Attributes that materially influence financial suitability,\n"
        "5) The existence of a financial account or relationship,\n"
        "6) Any reporting period or financial outcome that provides meaningful context.\n\n"
        "Prefer abstraction over deletion: preserve high-level financial meaning even when specifics must be removed."
    ),
    Domain.ECONOMIC: (
        "Focus on: macro or sector-level drivers, indicators, models, and causal reasoning. "
        "Drop: overly specific personal anecdotes or local identifiers that don't affect the "
        "analysis."
    ),
    Domain.POLICY_REGULATORY_COMPLIANCE: (
        "Focus on: obligations, rights, conditions, thresholds, exceptions, and enforcement "
        "mechanisms. Keep rule structure clear. Drop: individual case anecdotes unless they "
        "are essential exemplars."
    ),
    Domain.LEGAL: (
        "Focus on: (1) the core procedural arc of the case; (2) the legal basis for the "
        "application; (3) key domestic decisions and the reasons given; (4) treatment of "
        "detention, remand orders, extensions, and risks cited; (5) indictment, charges, "
        "and major trial-stage actions; (6) essential appeal outcomes; (7) any rights-"
        "interference events (e.g., censorship, lack of counsel). Always capture: "
        "(a) notification to the respondent Government and identification of Government "
        "representation; (b) the basic hearing chronology (at minimum the sequence or "
        "number of hearings, even if dates are abstracted); (c) any concurrent-sentence "
        "or sentence-merging orders; (d) the decision to join merits with admissibility "
        "or other major procedural consolidations. Drop: specific dates, place names, "
        "identities of individuals, and granular factual background unless they materially "
        "change the legal reasoning or procedural posture."
        "\n\n"
        "When the text is a legal form, authorization document, contract template, or "
        "procedural instrument, also treat the following as meaning-bearing elements that "
        "must be extracted in abstract form: "
        "- The presence of a 'customer information' or 'party information' section, "
        "captured abstractly (e.g., the form includes fields for the individual's identifying "
        "and account-related information). "
        "- The presence of financial or account-related fields needed to execute the "
        "authorization (e.g., payment instruments, account relationships, routing elements), "
        "expressed generically without repeating sensitive identifiers. "
        "- The structural components of the document such as required fields, placeholders, "
        "or sections (e.g., signature block, date field, merchant details, amount fields), "
        "when they define the meaning or operation of the document. "
        "- The fact that the form is a template requiring user-provided inputs (e.g., amount, "
        "date, merchant details), expressed abstractly."
        "\n\n"
        "Demographic quasi-identifiers (e.g., age, education level, nationality, ethnicity, "
        "gender) must be DROPPED unless they are CENTRAL to the legal reasoning, eligibility "
        "determination, statutory thresholds, contractual validity, or procedural rights. "
        "If such attributes appear but do not affect the legal meaning or enforceability of "
        "the document, they must NOT be turned into meaning units and must only appear, if at "
        "all, in highly abstract phrasing (e.g., 'general background information'), without "
        "preserving the specific attribute."
    ),
    Domain.HR_PEOPLE_OPS: (
        "Focus on: roles, responsibilities, performance dimensions, behavioral expectations, "
        "and process structure (e.g., review cycles). Drop: personal gossip, unnecessary "
        "names, and exact dates of incidents unless structurally important."
    ),
    Domain.MANAGEMENT_OPERATIONS: (
        "Focus on: goals, KPIs, processes, decision criteria, resource allocation, and "
        "coordination patterns. Drop: individual identities and low-level scheduling minutiae "
        "unless essential."
    ),
    Domain.CLINICAL_EHR_MEDICAL: (
        "Focus on: symptoms, diagnoses, assessments, clinical reasoning, interventions, timelines of care, "
        "and outcomes. The goal is to preserve the clinical story and decision-making logic while removing "
        "or abstracting details that could contribute to re-identification.\n\n"
        "Always keep, in appropriately abstract form:\n"
        "- Presenting symptoms, key past history elements that drive current assessment, and relevant exam findings,\n"
        "- The main working diagnoses or differential diagnoses (generalized when needed),\n"
        "- The clinician's reasoning about cause, risk, and prognosis,\n"
        "- Interventions, medications, and monitoring plans (prefer generic drug classes or roles over brand names "
        "unless the specific agent is critical to the clinical meaning),\n"
        "- High-level timelines of care (e.g., 'during childhood', 'over several years', 'at follow-up') rather than "
        "exact dates or ages.\n\n"
        "Clarification and education requests:\n"
        "- When a patient explicitly asks for explanations or definitions of medical, anatomical, imaging, or diagnostic "
        "terms (e.g., nerve root abutment, annular bulge or tear, MRI findings), you MUST preserve those terms in "
        "generalized clinical language rather than collapsing them into vague references.\n"
        "- Treat these as diagnostic-understanding or clinical-education intents, not as symptom reports. Capture both "
        "what concept the patient is asking about and how the clinician responds (explains, defers, or refers).\n\n"
        "Named conditions and findings:\n"
        "- Preserve non-identifying anatomical and pathological terms (e.g., nerve root contact, disc bulge, annular tear) "
        "when they are central to the patient's question or the clinician's reasoning.\n"
        "- Do NOT drop or over-generalize such terms if doing so would remove the substantive clinical concept or the "
        "educational value of the exchange.\n\n"
        "Short clinical exchanges:\n"
        "- In brief patient–clinician interactions (such as single-question consultations), prioritize capturing:\n"
        "  • The specific medical concepts or findings being asked about,\n"
        "  • The nature of the clinician's response (explanation, reassurance, deferral, or referral),\n"
        "  even if no formal diagnosis or complete treatment plan is provided.\n\n"
        "Benign viral conditions:\n"
        "- When benign, self-limited viral illnesses are mentioned (e.g., 'viral upper respiratory infection', "
        "'common cold', 'flu-like viral illness'), you MUST generalize them to a high-level phrase such as "
        "'a common viral illness' or 'a routine viral respiratory infection'.\n"
        "- Do NOT repeat the exact surface form from the sensitive entities block for these conditions, unless the "
        "specific named pathogen or syndrome is central to the clinical reasoning (e.g., a disease-specific treatment "
        "or public-health implication). In most primary-care-style counseling notes, abstraction to 'a common viral illness' "
        "is preferred.\n\n"
        "Handle sensitive attributes carefully:\n"
        "- Demographics (age, gender, ethnicity, detailed location) should be represented only in coarse categories "
        "(e.g., 'a young adult', 'an older adult') and only when they materially influence risk, diagnosis, or management.\n"
        "- Family medical history and genetic predispositions should be preserved only at the level needed for clinical "
        "logic (e.g., 'a family history of a related endocrine disorder') without copying exact surface forms.\n"
        "- Stigmatizing or highly sensitive conditions (e.g., certain mental health diagnoses, sexually transmitted "
        "infections, substance use) should be abstracted to category-level descriptions unless the specific label is "
        "necessary to understand the clinical decision-making.\n\n"
        "Abstract or drop:\n"
        "- Highly identifying combinations of demographics, rare conditions, and free-text anecdotes that are not "
        "essential to the clinical interpretation,\n"
        "- Exact dates, precise ages, and detailed timelines when a higher-level temporal description suffices,\n"
        "- Institution names, clinician names, and operational details that do not affect assessment or plan.\n\n"
        "When in doubt, preserve the clinical reasoning, the key medical concepts, and the care trajectory in generalized "
        "language, and prefer abstraction over deletion so that the medical meaning remains intact while sensitive surface "
        "forms are removed."
    ),
    Domain.EDUCATIONAL_PEDAGOGICAL: (
        "Focus on: learning objectives, knowledge structure, exercises, and evaluation "
        "criteria. Drop: specific student identifiers or one-off comments that do not "
        "change the instructional content."
    ),
    Domain.FICTION_CREATIVE: (
        "Focus on: key plot events, character arcs, world rules, and central themes. Drop: "
        "extraneous descriptive flourishes and minor side characters that do not affect "
        "story coherence."
    ),
    Domain.ENTERTAINMENT_MEDIA: (
        "Focus on extracting distinct information-bearing units about media content, formats, "
        "celebrity behavior, and cultural themes. Do NOT summarize or compress multiple ideas "
        "into one unit if they can be separated.\n\n"
        "Treat as separate meaning units when possible:\n"
        "• Each show, segment, or series mentioned (news specials, talk shows, documentaries, reality shows).\n"
        "• Each description of format or style (dramatic reconstructions, panel debate, visual style, tone).\n"
        "• Each opinion or evaluative judgment about a show, season, host, or guest.\n"
        "• Each notable celebrity appearance or interaction (who appears with whom, in what context, doing what).\n"
        "• Each cultural or social theme raised (gender roles, sexuality, family dynamics, fame, audience tastes).\n"
        "• Each programming or scheduling decision (special coverage, repeats tied to new releases, new feature launches).\n"
        "• Each observed trend in channel or industry behavior (early election coverage, increasingly suggestive content, etc.).\n\n"
        "Segmentation bias: when a sentence contains multiple shows, guests, format choices, or opinions, "
        "prefer splitting into multiple meaning units rather than merging them, as long as each unit still "
        "conveys a coherent, self-contained claim."
    ),
    Domain.SOCIAL_CULTURAL_OPED: (
        "Focus on: main argument, supporting points, evidence, and counterpoints. Drop: "
        "gratuitous personal attacks, identifying anecdotes about third parties, and "
        "unnecessary specifics that don't change the argument."
    ),
    Domain.PROCEDURAL_INSTRUCTIONAL: (
        "Focus on: step-by-step actions, prerequisites, warnings, and success criteria. "
        "Drop: personal narratives or identifiers that aren't needed to follow the procedure."
    ),
    Domain.META_TEXT: (
        "Focus on: labels, annotation schemas, instructions, and configuration semantics. "
        "Drop: incidental examples that leak real-world identifiers if they are not central "
        "to understanding the schema."
    ),
    Domain.SOCIAL_MEDIA: (
        "Focus on: core claim, sentiment, and key entities or events referenced at an "
        "abstract level. Drop: handles, hashtags that encode identities, and specific URLs "
        "unless structurally essential."
    ),
    Domain.TRANSCRIPTS_INTERVIEWS: (
        "Focus on: questions, answers, decisions, commitments, and key reasoning steps. "
        "Drop: greetings, small talk, and side chatter that do not influence outcomes."
    ),
    Domain.OTHER: (
        "When the text doesn't clearly fit any domain, use general judgment: keep the "
        "core purpose, main actions, decisions, and outcomes; drop hyper-specific identifiers."
    ),
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _get_domain_classification_prompt(data_summary: str | None = None) -> str:
    domain_descriptions = [f'- "{domain.value}":\n  {desc}' for domain, desc in _DOMAIN_LIST]
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
- If the text is an email/chat discussing a contract → "CHAT_EMAIL_CSAT", not "LEGAL".
- If the text is a narrative about a worker's life at a company → "BIOGRAPHY", not "MANAGEMENT_OPERATIONS".
- If the text matches two domains, choose the dominant purpose and structure.
</disambiguation_guidelines>

<text_to_classify>
<<TEXT>>
</text_to_classify>"""
    return (
        prompt.replace("<<TEXT>>", _jinja(COL_TEXT))
        .replace("<<DOMAINS>>", domains_text)
        .replace("<<DATA_CONTEXT>>", data_context_section)
    )


# ---------------------------------------------------------------------------
# Custom column generator
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_DOMAIN])
def _enrich_domain(row: dict[str, Any]) -> dict[str, Any]:
    """Look up domain-specific guidance from DOMAIN_SUPPLEMENT_MAP."""
    parsed_domain = DomainClassificationSchema.model_validate(row[COL_DOMAIN])
    row[COL_DOMAIN_SUPPLEMENT] = DOMAIN_SUPPLEMENT_MAP[parsed_domain.domain]
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
            LLMStructuredColumnConfig(
                name=COL_DOMAIN,
                prompt=_get_domain_classification_prompt(data_summary),
                model_alias=domain_alias,
                output_format=DomainClassificationSchema,
            ),
            CustomColumnConfig(
                name=COL_DOMAIN_SUPPLEMENT,
                generator_function=_enrich_domain,
            ),
        ]
