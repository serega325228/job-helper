# Job Helper

## Runtime configuration

The frontend supplies LLM connection details, generation parameters, and feature
switches. They are persisted in `data/config.json` (under `Settings.app.data_dir`).
`LLMConfigManager` validates and loads that file into `LLMSettings.config` and
`Settings.features` at startup. These runtime values are excluded from environment
loading. An absent file leaves both sections unconfigured; there are no fallback
model, token-budget, temperature, or feature values.

Updates use one process-local lock and an atomic file replacement. Runtime settings
change only after saving succeeds. Each HTTP request receives its own `LLMProvider`
with a copy of the configuration and credentials, so changing settings does not
reroute an in-flight request. Run one backend process for this file-based store;
multiple workers require shared transactional storage.

Provider keys are encrypted using the existing `SecurityService`; back up
`data/.secret_key` with the configuration file. Invalid files and unreadable keys
raise errors without overwriting saved data. Configuration responses mask keys.

The following paths are relative to the API prefix:

- `POST /config/api-keys`: set provider keys; an empty string deletes a key.
  `gemini` is canonical; `google` remains an accepted input alias.
- `GET /config/llm-api-key`: read the masked configuration, or `null` before setup.
- `PUT /config/llm-api-key`: replace the complete LLM configuration. Credentials
  come from the provider-key store. Local providers may run without credentials.
- `GET /config/features`: read feature settings, or `null` before setup.
- `PUT /config/features`: replace all four feature switches.
- `POST /config/llm-test`: test saved settings, or a complete candidate body with
  an optional `api_key` override. The structured-output probe never saves or
  activates the candidate. Saving configuration does not make a paid test call.

Example LLM input (the numbers here are submitted values, not server defaults):

```json
{
  "provider": "ollama",
  "model": "your-installed-model",
  "api_base": "http://localhost:11434",
  "max_tokens": 4096,
  "temperature": null,
  "reasoning_effort": null,
  "api_version": null
}
```

`temperature` is required: use a number from 0 to 2, or `null` to omit the parameter.
Ollama, OpenAI-compatible servers, and Azure Foundry require an explicit endpoint.
Other providers use their standard endpoint when `api_base` is null. Provider
credentials and endpoints do not fall back to environment variables. Unknown
models can be saved and tested; known unsupported parameters and token limits
are rejected. Timeouts and retry counts remain server-owned `LLMSettings` fields.

Example feature input:

```json
{
  "enable_cover_letter": true,
  "enable_outreach_message": false,
  "enable_interview_prep": true,
  "allow_unverified_skills": false
}
```

Incomplete or invalid input returns 422. Operations requiring missing configuration
return `409 configuration_required`. Feature switches must be actual JSON booleans.
Language, tailoring-strategy, and custom-prompt endpoints remain available. Custom
cover-letter/outreach templates must contain `{job_description}`, `{resume_data}`,
and `{output_language}` and cannot use other placeholders or attribute access.
The legacy database-reset endpoint returns 501 after confirmation; its database
implementation remains outside this refactor.

## Resume generation

`ImproverService`, `RefinerService`, `InterviewPrepService`, and `CoverLetterService`
accept `ResumeData` rather than arbitrary dictionaries. Pass a `FeatureConfig`
snapshot explicitly into operations that depend on feature switches. Keyword
extraction returns `JobKeywords`; structured LLM output is validated by the provider.

Improvement proposes targeted edits and applies only permitted action/path pairs.
Identity fields, dates, metadata, and section membership are preserved. Reorders
must preserve every item and duplicate. Keyword refinement uses the same patch
validator. Full-resume regeneration and the separate LLM skill-planning pass are
removed.

Results include applied and rejected changes, warnings, and typed improvement
suggestions. Applied suggestions describe accepted edits; recommendations describe
remaining gaps. Nothing is reported as improved when no change occurred.

**Allow unverified skills** permits adding skills explicitly mentioned in the job
description without evidence in the master resume. These additions are labeled
unverified and survive refinement under the same policy. They cannot authorize
invented work history, qualifications, or metrics. Normal mode requires resume
evidence. Evidence checks use skill aliases and term matching, not a semantic proof
of proficiency; metric warnings and edit review still matter.

## Vacancy matching

Vacancy processing uses Taskiq and RabbitMQ. The collection and matching
LangGraph workflows have been removed; resume and supervisor agents remain separate.

Detailed processing persists raw details, normalizes them, creates embeddings,
and selects the existing top `rerank_limit` candidates (40 by default, with
`search_limit=100`). There was no percentage setting in the previous matcher.
Only that shortlist is reranked. Every successfully parsed vacancy receives a new
Laya evaluation using its full description and parsed conditions, including
vacancies outside the shortlist. Those vacancies finish as `filtered`; shortlisted
vacancies receive final scores and a `VacancyMatch` and finish as `completed`.

Laya's `none`, `weak`, `good`, and `strong` labels map to `0`, `1/3`, `2/3`,
and `1`. Existing final geometric score weights and match categories are retained.
A zero component makes the combined score zero. Labels and numeric scores are
saved in `component_scores.laya`; full vacancy rows also retain their evaluation
and final scores. These ordinal values need calibration against labeled matches.

## Embedding and reranking backends

`Embedder` and `Reranker` protocols in `src/ports` let services use either backend.
Both default to `llama_server`. The previous `EmbeddingService` lives in
`src/infrastructure/embedder/local.py`; the previous `VacancyReranker` remains
in `src/infrastructure/reranker/local.py`. Local libraries are imported
only when their backend is selected. DI owns and closes the HTTP clients and the
local embedding model.

Configure independent servers in `.env` (URLs are server roots, without `/v1`):

```dotenv
EMBEDDING_BACKEND=llama_server
EMBEDDING_BASE_URL=http://localhost:8081
EMBEDDING_SERVER_MODEL_NAME=Qwen3-Embedding-0.6B-Q8_0.gguf
RERANKER_BACKEND=llama_server
RERANKER_BASE_URL=http://localhost:8082
RERANKER_SERVER_MODEL_NAME=Qwen3-Reranker-0.6B-Q8_0.gguf
```

Each backend also accepts `*_API_KEY` for optional Bearer authentication,
`*_REQUEST_TIMEOUT_SECONDS` (default 120), and `*_BATCH_SIZE` (default 16).
HTTP connection pools live in the Dishka `hh`, `embedding`, and `reranker`
components. Each client uses its own service settings, stays cached at
`Scope.APP`, and is closed by its async generator when the container closes.
Consumers select their client with
`Annotated[httpx.AsyncClient, FromComponent("hh")]` (or `"embedding"`/`"reranker"`).
HH's timeout no longer controls clients for the model services.
Nested `EMBEDDING__...` and `RERANKER__...` settings are also supported. When the
application runs in Docker, use server container hostnames and their internal
ports instead of `localhost`.

For example, run the two CPU servers with the official llama.cpp Docker image:

```sh
docker run --rm -p 8081:8080 -v "$PWD/.models:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --alias Qwen3-Embedding-0.6B-Q8_0.gguf \
  --embedding --pooling last --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --batch-size 8192 --ubatch-size 8192 --parallel 1

docker run --rm -p 8082:8080 ghcr.io/ggml-org/llama.cpp:server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF \
  --alias Qwen3-Reranker-0.6B-Q8_0.gguf \
  --embedding --pooling rank --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --batch-size 8192 --ubatch-size 8192 --parallel 1
```

The adapters use llama.cpp's documented [`/v1/embeddings` and `/v1/rerank`
endpoints](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
Both HTTP adapters use `httpx.AsyncClient`. Embedders expose one generic async
`embed(texts)` method and receive already prepared text. `ScoringService` applies
Qwen's `Instruct: ...\nQuery: ...` format for job searches; `VacancyService` combines
vacancy titles and content without a query instruction. Embeddings use all 1024
components without truncation. The shared `EMBEDDING_DIMENSIONS` constant lives
in `src/ports/embedder.py` and also defines the database vector column widths.
Reranking restores input order from response indexes and preserves
Qwen's scores in the 0–1 range. Invalid response indexes, dimensions, vectors,
and scores fail before persistence; HTTP failures propagate to callers.

To select the previous implementations independently, set `EMBEDDING_BACKEND=local`
with `EMBEDDING_MODEL_PATH` pointing to its GGUF, and/or `RERANKER_BACKEND=local`
with `RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3`. Local inference runs in a worker
thread with its existing model lock; prompt preparation belongs to services.
The local embedding model must also produce 1024 components.
Existing databases with 768-component vector columns need a schema migration to
`vector(1024)` for both title and content in `vacancies` and `preference_intents`;
changing the Python models does not alter existing tables.
Switching embedding dimensions, models, or prompt formats
requires regenerating **all** vacancy and preference embeddings before matching;
equal dimensions do not make vectors from different models comparable.

Run the HTTP adapter checks without running Docker or downloading models:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m unittest discover -s tests -p test_llama_server.py
```

## Vacancy processing workers

Run PostgreSQL with pgvector and RabbitMQ, configure the existing `DB_` settings
and `TASKIQ_RABBITMQ_URL`, and apply the additive schema upgrade before deploying
the new API or workers. The upgrade creates missing tables and adds pipeline
columns to existing vacancies; existing vacancy data is retained.

```sh
.venv/bin/python -m src.infrastructure.db.vacancy_pipeline
.venv/bin/uvicorn src.main:app
.venv/bin/taskiq worker src.tasks.scraping:io_broker --ack-type when_executed --workers 2 --max-async-tasks 5
.venv/bin/taskiq worker src.tasks.scraping:laya_broker --ack-type when_executed --workers 1 --max-async-tasks 1
.venv/bin/taskiq scheduler src.tasks.scraping:scheduler
```

All Laya evaluations run on the `laya` queue. Collection, keyword filtering,
detailed scraping, normalization, embedding scoring, reranking, final persistence,
and reconciliation run on `io`. Each worker opens the other broker for publishing
the next stage. Each queue has its own durable TTL retry queue, so delayed retries
return to the correct worker without a RabbitMQ plugin.
[Taskiq acknowledgement documentation](https://taskiq-python.github.io/guide/cli.html)
describes `when_executed`, which is configured on every task and in these commands.

Start collection with `POST /api/v1/vacancies/previews/collect`:

```json
{
  "profile_id": "PROFILE_UUID",
  "sources": ["linkedin", "hh"],
  "preferences": ["Backend"],
  "query": {"text": "Python backend", "location": "Germany"},
  "hard_filters": {"work_formats": ["remote"]}
}
```

Either names list can be `["all"]`. The API resolves enabled preference IDs and
source IDs before enqueueing collection IDs. Existing `companies` rows represent
sources: `name` is the selectable source name, `careers_url` is its listing page,
and `search_settings.source` names its adapter (`linkedin` or `hh`).
Built-in source rows are created when first selected. An optional
`vacancy_page_urls` mapping overrides stored URLs for the selected sources.

Listing pages are scraped in batches and previews are committed immediately
after duplicate checking, before any evaluation task is published. Exact
`(source, external_id)` matches are skipped. Normalized company/title matches
against previews or vacancies discovered in the previous seven days are treated
as recent reposts. Initial inserts use uniqueness constraints and conflict-safe
inserts; advisory transaction locks also serialize competing repost checks.
Later stages update the same row.

Descriptions of at least `SCRAPING_PREVIEW_DESCRIPTION_MIN_LENGTH` characters
(default 80) are eligible for early Laya evaluation. This is a configurable
sufficiency heuristic. Smaller previews use keyword relevance against the selected
preferences. Known hard constraints and required/excluded keywords apply to both
paths; unknown preview fields remain eligible. Preview evaluation is persisted
and only gates collection readiness.

Read previews with `GET /api/v1/vacancies/previews/PROFILE_UUID`, then start details
with `POST /api/v1/vacancies/process`:

```json
{
  "profile_id": "PROFILE_UUID",
  "preview_ids": ["PREVIEW_UUID"],
  "hard_filters": {"work_formats": ["remote"]},
  "role_fit": ["good", "strong"],
  "skill_fit": ["weak", "good", "strong"]
}
```

Omit `preview_ids` to select ready previews by filters. Laya label filters apply
when a preview has an evaluation. The API creates the initial vacancy rows and a
persistent scoring batch atomically, then publishes vacancy IDs for detailed
scraping. That batch keeps the embedding shortlist consistent across parsing
chunks. The tasks publish their successors directly.

Every processing task reloads and locks its existing entities and checks the
expected status. Completed stages exit without repeating work. Results and next
statuses commit in the same transaction before publishing the successor.
Row locks are held through bounded stage I/O to prevent concurrent execution;
size database pools and worker concurrency accordingly. Worker death rolls back
the stage transaction and releases its locks.

The scheduled reconciliation task is the only global unfinished-work scan.
It selects stale non-terminal rows with `FOR UPDATE SKIP LOCKED`, refreshes their
scheduling timestamp atomically, commits, then republishes the appropriate stage.
A commit/publish gap, exhausted stage retries, or a failed recovery publication
is retried after `TASKIQ_STALE_TIMEOUT_SECONDS` (default 900). Concurrent recovery
runs and currently executing stages cannot claim the same locked row. Run one
scheduler; `TASKIQ_RECOVERY_CRON` defaults to once per minute.

Taskiq owns stage retries: `TASKIQ_STAGE_ATTEMPTS=3`, with backoff and jitter
starting at `TASKIQ_RETRY_DELAY_SECONDS=5`.
HTTP/model clients perform one or two attempts, and normalization has no extra
business retry loop. RabbitMQ redelivery handles unacknowledged messages after
worker or connection loss.

Install Chromium with `.venv/bin/playwright install --with-deps chromium`.
HeadHunter uses its official API; LinkedIn uses public job pages and does not
authenticate or bypass challenges. `CRAWLEE_` settings control concurrency
(1/3/5), request rate (30/minute), retries (1), and navigation/handler timeouts
(30/300 seconds). `SCRAPING_` settings control listing pages (10), previews (250),
details (50), normalization batch size (5), and Laya batch size (32).
Crawlee temporary storage remains isolated per invocation.

## Checks

Run the standard-library suite without contacting LLM providers:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m unittest discover -s tests
```

The configuration and health routers are exercised in an isolated FastAPI app.
Vacancy checks cover task replay, batch completion, shortlist ordering, fresh
detailed evaluation, preview persistence before publication, repost detection,
and stale recovery. They use fake clients and database sessions; a live
PostgreSQL/RabbitMQ deployment is needed to verify transport and database locking.
