# Job Helper — Design Decisions

This file replaces the previous README. It records chosen approaches, their
reasons, and proposals awaiting agreement. It is not a claim that the described
features are implemented.

## Status conventions

- **Accepted**: explicitly agreed direction.
- **Proposed**: recommendation, not yet an implementation commitment.
- **Open**: needs a product decision or user clarification.

## Product direction — described by the user

A local AI-assisted job-search and application workspace. Users connect an LLM
through LiteLLM. Local hosting does not necessarily mean local inference.

Intended workflows:

- Import an existing master resume, or tell a personal/professional story, to
  build a profile. Suggest preferences from that material for user review.
- Search aggregators using preferences and additional filters; inspect previews
  before paying the cost of detailed scraping and evaluation.
- Show up to 50 suitable vacancies, with explanations of suitability and gaps.
- Accept pasted vacancy text through the same normalization/evaluation pipeline.
- Improve a master resume or tailor a resume, cover letter, outreach message, and
  interview preparation to a vacancy; allow manual review and template-based PDF
  export, including Russian-market layouts.

Ideas under consideration, not settled implementation choices:

- Embedding retrieval, reranking, and Laya evaluation, and which are necessary.
- Three editing modes: light rewording, enrichment from profile evidence, and a
  requested experimental mode that permits invented/changed facts.
- Multilingual style refinement, Crawl4AI for company sites, and a kanban-like
  application tracker. Automatic submission is not yet a specified workflow.

## D001 — Keep a decision log here

**Status: Accepted.**

Record each decision with its status, approach, reasons, and important trade-offs.
Keep proposals distinct from implemented behavior. Do not describe unimplemented
endpoints, filters, or deployment commands as working features.

**Reason:** the previous README had drifted from the code, making it an unreliable
source of truth.

## D002 — Natural-language preferences become validated drafts

**Status: Proposed.**

Users can enter preferences such as:

- `AI engineer`
- `ML engineer`
- `Golang engineer $400k/year FAANG`

Use this flow:

```text
User text
  -> LLM structured extraction
  -> deterministic validation and taxonomy/company resolution
  -> editable preview with assumptions and clarification questions
  -> user confirmation
  -> application service persists the preference
```

The LLM returns a strict Pydantic draft, not ORM objects, SQL, or database writes.
It can select existing taxonomy IDs or flag an unresolved role; it cannot silently
create canonical roles. Validate IDs, enum values, monetary amounts, and constraint
semantics in application code. Treat the input as data, not instructions.

Store the original text, confirmed structured interpretation, and interpretation
version. Keep assumptions/questions in the draft; do not activate unresolved hard
constraints. Ordinary edits should not require another LLM call unless the user
asks to reinterpret the text.

**Reasons:** free text is convenient, structured fields are filterable, and user
confirmation prevents invented requirements from becoming invisible filters.
An LLM is useful for interpretation, not as the authority on user intent.

## D003 — Separate roles, skills, seniority, and conditions

**Status: Proposed.**

A preference is a search intent, not just a job title. Its conceptual shape is:

- Identity: ID, profile ID, display label, original text, enabled flag, weight.
- Role targets: canonical role IDs and optional explicit title phrases.
- Specialization: required/preferred skills, such as Go or Python.
- Seniority: optional bounds, separate from the role identity.
- Conditions: compensation, location, work format, employment type, companies.
- Policy: which conditions are hard requirements versus ranking preferences.
- Unresolved meaning: retained in the draft until clarified.

For example, `Golang engineer` usually suggests a backend/software engineering
role with Go specialization. Do not create a new canonical role for every
language, seniority, company, and salary combination. Show this interpretation
for confirmation; a Go role is not necessarily backend work.

Do not populate the candidate's demonstrated skills from desired job preferences.
A user can want an ML job without already having every ML skill.

**Reasons:** role identity stays stable, the title catalog stays manageable, and
skills/conditions can be evaluated independently instead of being hidden in text.

## D004 — A small, curated, versioned title taxonomy

**Status: Proposed.**

Start with a checked-in catalog, for example `data/job_roles.json` (not yet
created). Load it through one taxonomy service shared by preference parsing,
source-query generation, vacancy normalization, and autocomplete.

Each role should have:

- Stable ID, e.g. `ai_engineer`, `ml_engineer`, `backend_engineer`.
- Display label and a short definition.
- Curated aliases, with locale where relevant.
- Explicit related roles, which are not automatically equivalent.

Example distinction:

| Role ID | Display label | Example aliases |
| --- | --- | --- |
| `ai_engineer` | AI Engineer | Artificial Intelligence Engineer |
| `ml_engineer` | Machine Learning Engineer | ML Engineer |
| `backend_engineer` | Backend Engineer | Backend Developer, Back-end Engineer |

These are seed examples, not a complete occupational classification. AI Engineer
and ML Engineer are related roles, not guaranteed synonyms. Broad terms such as
`Software Engineer` may remain ambiguous until a description is available.

Keep title aliases separate from skill aliases (`Golang` -> `Go`). A title such
as `Senior Golang Backend Engineer` can suggest role `backend_engineer`, seniority
`senior`, and skill `Go`; preserve the original title and evidence for each hint.
A title can suggest multiple roles rather than being forced into exactly one.

Use deterministic normalization and curated phrase/token matching first. Account
for punctuation and language-specific boundaries; do not use unrestricted
substring matching. Optionally use embeddings for uncertain cases after measuring
whether they improve recall. Use the LLM to suggest catalog changes from unmatched
titles in batches, with review before publishing them.

**Reasons:** stable IDs support reproducible matching and user-visible labels.
A curated catalog prevents spelling variants and LLM inventions from fragmenting
roles. A file is sufficient for an MVP; use database-backed catalog management
only if live editing or multiple maintainers make it worthwhile. ESCO/O*NET may
later supply external mappings, but are not required to start.

## D005 — Retrieval is broad; final matching is evidence-based

**Status: Proposed.**

Distinguish three concepts:

1. **Source queries:** strings/filters supported by a job board, used to retrieve
   candidates. Build them from role aliases and relevant specialization terms.
2. **Preview classification:** cheap title/snippet/company checks to decide which
   descriptions are worth fetching.
3. **Detailed evaluation:** normalized job facts evaluated against confirmed
   constraints, followed by ranking.

Scrape listing previews first, persist/deduplicate them, then decide which detail
pages to fetch. Do not require exact canonical-title equality. Apply title matching
as a high-recall gate:

- **Likely match:** known role/title alias and compatible visible qualifiers.
- **Uncertain:** broad, unfamiliar, or underspecified title; keep eligible for a
  bounded detail-fetch/review budget instead of silently rejecting it.
- **Mismatch:** clear incompatible role or a confirmed exclusion supported by
  available evidence.

Do not make an LLM call for every preview in the initial implementation. Use the
same normalized role representation on preferences and vacancies. An embedding
score is a ranking signal, not proof that a hard requirement passed.

**Reasons:** titles vary across employers, and previews often omit salary,
technology, seniority, and even useful role details. Exact-title-only filtering
loses relevant jobs; full-description scraping of every result wastes resources.

## D006 — Preserve alternatives and uncertainty explicitly

**Status: Proposed.**

Separate preference entries are alternatives (OR). Within one preference,
confirmed hard conditions combine with AND. Role targets within one preference
are alternatives unless the user explicitly asks otherwise. Soft conditions
influence ranking rather than eligibility.

For example:

```text
AI engineering preference
OR ML engineering preference
OR (confirmed Go role AND confirmed compensation rule AND confirmed company rule)
```

Evaluate hard conditions using `pass`, `fail`, or `unknown`:

- Reject a confirmed hard mismatch.
- Missing evidence is unknown, not a failure or an automatic pass.
- Unknown preview facts remain eligible for detailed evaluation.
- Unresolved facts after detail processing remain visibly unknown; never label
  them as satisfying all requirements. Support an explicit strict-results view.

Persist the preview/vacancy-to-preference association and match reasons. One job
may match several preferences. Fetch and normalize it once, then evaluate each
preference separately. A UI may choose a best match without discarding the others.

**Reasons:** global filters must not accidentally turn alternatives into a single
impossible query, and one early winning preference must not hide another valid
match. Explanations make both false positives and false negatives diagnosable.

## D007 — Compensation and company groups need explicit meaning

**Status: Proposed; interpretation of the example remains open.**

For `Golang engineer $400k/year FAANG`, the draft can extract Go, the amount
400,000, annual period, and the company-group phrase. It should ask:

- Is $400k a minimum, target, or approximate range?
- Which dollar currency is intended? USD can be suggested, not silently asserted.
- Is this base salary, cash compensation, or total compensation including equity?
- Is it gross or net, and is compensation a hard condition or a preference?
- Does FAANG mean its literal members or a broader group of large tech companies?

Represent compensation with amount/range, currency, period, basis, tax treatment,
and hard/soft policy. Unknown basis stays unknown. Do not compare base salary
against total compensation or currencies without an explicit conversion policy.
Do not invent an annualization rule for hourly compensation.

Represent employer groups separately from role titles. A curated FAANG group can
resolve employer aliases to stable company IDs (Meta/Facebook, Alphabet/Google,
Amazon, Apple, Netflix). Display its membership during confirmation. Never equate
"big tech" with that exact group without asking.

**Reasons:** otherwise the parser can turn an ambiguous sentence into a precise
but wrong database filter and reject nearly every vacancy.

## D008 — Separate candidate facts from resume documents

**Status: Proposed.**

Treat the profile as the confirmed fact base; master and tailored resumes are
versioned presentations of it, not independent sources that silently overwrite
one another. Imported resumes and stories create candidate facts with source
references and review status. Model inferences remain distinguishable from user
statements and confirmations. Preference suggestions require confirmation too.

A profile may contain more detail than any one resume. Missing content in an
import is not evidence that an existing fact should be deleted. Conflicts and
material changes should be reviewed. Do not promote generated resume statements
back into confirmed profile facts automatically.

Keep an explicit resume-profile revision link and a way to review intentional
manual edits; one fact base must not become an excuse to overwrite user drafts.

**Reasons:** otherwise inference and generated claims can become their own
"evidence", successive tailoring passes drift, and importing a short resume loses
useful candidate history.

## D009 — Suitability, document alignment, and parsing quality are different

**Status: Proposed.**

Show separate dimensions rather than one authoritative "ATS percentage":

- Job suitability: preference constraints and candidate qualifications, evaluated
  from the confirmed profile and vacancy facts.
- Resume-to-job alignment: how well this document expresses relevant evidence.
- Document parseability: reading order, text extraction, sections, and formatting.

None is a validated probability of passing an employer's ATS or receiving an
interview. Internal scores should expose components, missing facts, and reasons.
Tailoring can improve document alignment without improving real qualifications.

For before/after comparisons, freeze the vacancy revision, profile revision,
scorer/prompt/model versions, and scoring configuration. Re-evaluate both resume
versions under that same setup. Record absolute alignment scores separately from
cohort-relative ranking; changing the search cohort must not masquerade as a
resume improvement. Language changes and model nondeterminism also affect scores.

**Reasons:** keyword stuffing or invented qualifications can game self-evaluation,
and changing models or baselines produces misleading improvement claims.

## D010 — One ingestion pipeline, progressively more expensive evaluation

**Status: Proposed.**

All adapters, including pasted vacancy text, converge on a versioned raw vacancy
and one normalized vacancy schema. Manual input bypasses discovery, not validation
or evaluation. Preserve source text and distinguish unknown from extracted facts.

Recommended baseline:

```text
Search queries / company listings
  -> persist and deduplicate previews
  -> cheap role relevance + known hard constraints
  -> bounded detailed scraping
  -> normalization and confirmed constraint checks
  -> inexpensive structured/profile and resume-alignment assessment
  -> optional semantic reranking / Laya
  -> persist explanations and show up to 50 results
```

Embeddings, reranking, and Laya are independent optional stages until evaluation
shows their incremental value. Embeddings are useful when a large stored corpus
needs cheap semantic retrieval; an ANN/vector stage is not a prerequisite for
scoring a small fresh batch. Add reranking or Laya separately, measuring relevance,
recall loss, latency, memory, and inference cost against a simple baseline. A top-K
cut before a later model is a recall trade-off and must be measured.

Do not pad the list with unsuitable jobs to reach 50. Preserve results outside the
first page and rejection reasons for inspection. Support cancellation, per-run
budgets, retries, and stage replay without duplicate work.

Keep source integrations behind adapters. Prefer stable official APIs and ATS
feeds when available. Evaluate Crawl4AI as a fetching/extraction implementation
for company pages, not a replacement for identity validation, pagination, dedup,
filtering, or job-fact normalization. Reuse existing browser infrastructure until
there is a concrete source that needs another tool.

**Reasons:** a local app has limited resources, overlapping models may not improve
results, and adapters should not produce incompatible downstream behavior.

## D011 — Reviewable editing and multilingual style refinement

**Status: Proposed; the requested fabrication mode remains an open product decision.**

Suggested trustworthy editing modes:

1. Polish: wording, ordering, clarity, and compression; no new factual claims.
2. Evidence-backed enrichment: include relevant facts from the confirmed profile
   that were absent from the current master/tailored resume.
3. Aggressive presentation: larger structural changes and stronger positioning,
   still grounded in facts, with questions to collect missing evidence.

The user also requested considering a mode that can invent/change facts. This is
not equivalent to enriching from profile evidence: it creates false claims, may
harm applications, and invalidates suitability comparisons. If an experimental
fictional draft is supported, isolate and label its claims; never promote them
into confirmed profile/master facts, silently merge them into truthful drafts,
or present score gains as qualification gains. Export behavior remains open.

Every mode should produce an editable diff with factual additions highlighted,
source references where available, and explicit user approval before replacing a
master resume or exporting a final application document.

Do not equate generic words with proof of AI authorship. Make style review
language-aware and optional: deterministic rules for clear repetition/formatting,
with an optional constrained LLM rewrite for fluency. Protect names, dates,
metrics, credentials, and technical terms; review any changed claims. No style
pass should promise that a document is "AI-undetectable".

**Reasons:** aggressive blacklist deletion damages natural language, separate
lists for every language are brittle, and an LLM rewrite can introduce new facts.

## D012 — Version exports and application history

**Status: Proposed.**

Offer a small set of ATS-readable templates first, including a Cyrillic-capable
A4 layout. Embed fonts, test text extraction and reading order, and support both
Russian and English content. Locale affects labels, dates, and typography; photos
and additional personal details should be optional, not universal requirements.
Validate output with representative long resumes, not screenshots alone.

An application tracker can start with saved -> preparing -> applied -> interview
-> offer/rejected/withdrawn, plus notes, links, dates, and reminders. An application
links to an immutable submitted resume/cover-letter version and vacancy snapshot.
The current vacancy and current master resume may change independently. Allow
multiple application attempts without overwriting history. Defer automatic
submission until review, external actions, and failure handling are specified.

**Reasons:** users need to know what they actually sent, and a ranking list alone
does not manage a job search. A basic tracker is often more useful than adding
another scoring model.

## Cross-cutting risks to address before expanding scope

- Resumes and personal stories are sensitive. Explain which text is sent to which
  remote provider, redact logs, protect keys, and allow data deletion/backups.
- Localhost access is not a complete security boundary: restrict administrative
  actions and network destinations. Treat fetched HTML and pasted vacancy text as
  untrusted data, including prompt injection, unsafe links, and frontend markup.
- Job sources change, throttle requests, and may restrict automated access.
  Respect source policies and access controls; make partial failures observable.
- Audit reused Resume-Matcher code/assets and their actual license/attribution
  obligations before redistributing; do not assume copying grants unrestricted use.
- Current startup/schema/DI failures must be repaired before layering on features.

## Preference implementation sequence if these proposals are accepted

1. Repair current preference model/import/relationship inconsistencies first.
   The model uses `general_title`, while several consumers/constraints still
   expect `name` or `PreferenceIntent`; the current representation is not usable.
2. Define strict draft/confirmed preference schemas and constraint policies.
3. Add the small role catalog, resolver, and company-group definitions.
4. Implement extraction -> clarification -> confirmation -> persistence.
5. Replace generic title-token filtering with shared role/qualifier matching.
6. Preserve per-preference matches, evidence, and unknown conditions.
7. Add fixture-based evaluation before introducing semantic fallbacks: AI vs ML,
   Go vs Golang, broad Software Engineer titles, multilingual aliases, seniority,
   missing salary, compensation basis, employer aliases, and OR/AND behavior.

Measure preview recall and detail-fetch cost on labeled examples; do not choose
semantic thresholds from intuition or treat LLM confidence as calibrated accuracy.

## Open product decisions

- Initial supported languages, role coverage, and job sources.
- Whether users want separate AI/ML preferences or one preference with both roles.
- Whether related-role expansion is opt-in (recommended) or a default.
- Default handling of unknown facts in the final results view.
- Compensation defaults and whether company groups can be user-defined.
