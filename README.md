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

The matching graph searches by preferences, compares and reranks candidates, then
saves matches. Comparison and reranking share one load of the profile, vacancies,
and preferences. Hard constraints are checked before profile scoring or model
inference; only the top `rerank_limit` candidates reach the reranker.

Laya evaluates role and skill fit for each vacancy's selected preference. Its
`none`, `weak`, `good`, and `strong` labels map to `0`, `1/3`, `2/3`, and `1`.
Each fit contributes 10% to both the shortlist and final geometric scores. A zero
component makes the combined score zero. These ordinal values are uncalibrated;
calibration requires labeled matches. Labels and numeric scores are saved in
`component_scores.laya` with matcher version `structured-laya-rerank-v1`.

## Embedding and reranking backends

`Embedder` and `Reranker` protocols in `src/ports` let services use either backend.
Both default to `llama_server`. The previous `EmbeddingService` lives in
`src/infrastructure/embedding/embedding.py`; the previous `VacancyReranker` remains
in `src/infrastructure/reranker/vacancy_reranker.py`. Local libraries are imported
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
The embedder applies Qwen's `Instruct: ...\nQuery: ...` query format; documents have
no instruction. [Qwen3-Embedding-0.6B supports reduced embedding
dimensions](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), so the adapter keeps
the first 768 components and normalizes them again to match the existing database
columns. Reranking restores input order from response indexes and preserves
Qwen's scores in the 0–1 range. Invalid response indexes, dimensions, vectors,
and scores fail before persistence; HTTP failures propagate to callers.

To select the previous implementations independently, set `EMBEDDING_BACKEND=local`
with `EMBEDDING_MODEL_PATH` pointing to its GGUF, and/or `RERANKER_BACKEND=local`
with `RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3`. The local embedding prompts and
pooling behavior remain unchanged. Switching embedding models or prompt formats
requires regenerating **all** vacancy and preference embeddings before matching;
equal dimensions do not make vectors from different models comparable.

Run the HTTP adapter checks without running Docker or downloading models:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m unittest discover -s tests -p test_llama_server.py
```

## Browser vacancy collection

`VacancyScrapingService.scrape()` runs the browser pipeline independently of HTTP
handlers or Celery. The initial adapter supports LinkedIn's public job search;
HeadHunter continues using the existing official API source. Crawlee manages
browsers, concurrency, request scheduling and retries. Listing cards are parsed
in bulk with Selectolax Lexbor after rendering; only selected previews become
`DETAIL` requests.

Call it from an async application entry point:

```python
from uuid import UUID

from src.di.container import create_container
from src.infrastructure.vacancy_sources.linkedin.source import LinkedInVacancySource
from src.schemas.vacancy import VacancyHardFilters, VacancyScrapingQuery
from src.services.vacancy_scraping import VacancyScrapingService

async def collect(profile_id: UUID):
    async with create_container() as container:
        service = await container.get(VacancyScrapingService)
        source = await container.get(LinkedInVacancySource)
        return await service.scrape(
            source,
            VacancyScrapingQuery(text="Python backend", location="Germany"),
            profile_id,
            filters=VacancyHardFilters(work_formats=["remote"]),
        )
```

Install Chromium and its Linux system libraries with
`.venv/bin/playwright install --with-deps chromium` in the deployment image.
The database schema, a stored profile,
Laya model, existing normalization LLM configuration, and existing embedding model
must be available. No new Python dependencies are required. Embeddings are used only by
the existing persistence/matching flow, never to select previews.

Deterministic checks use explicit known facts and enabled preferences; unknown
fields remain eligible. `required_keywords` and `excluded_keywords` optionally
constrain preview text. Laya batches reuse the existing role/skill questions;
role fit must be `good` or `strong` and skill fit must be at least `weak`.
Full details go through `RawVacancy`, the existing normalizer, `NormalizedVacancy`,
and the existing vacancy repository/unit of work. DI providers supply factories
that open separate request scopes so concurrent handlers never share a database
session. The scraping service receives its dependencies and pipeline limits
explicitly; it does not access the Dishka container or application settings.
`CrawleeProvider` owns per-run crawler construction, temporary storage, and cleanup.

`CRAWLEE_` environment settings control concurrency (1/3/5), request rate
(30/minute), retries (2), navigation/handler timeouts (30/300 seconds), and headless
mode. `SCRAPING_` settings control search pages (10), unique previews (250), details
(50), and Laya batch size (32). Move previous browser-related `SCRAPING_` keys to
`CRAWLEE_`, for example `SCRAPING_MAX_CONCURRENCY` becomes `CRAWLEE_MAX_CONCURRENCY`.
Nested `CRAWLEE__...` and `SCRAPING__...` settings follow the application's convention.
The native load-more loop stops at its configured limits or the site's end marker.
Temporary Crawlee storage is isolated per invocation; committed vacancies provide
deduplication by source/external ID and URL on subsequent runs. An interrupted
process starts a fresh discovery run; its temporary queue is not a resume store.
Search retries retain evaluated previews, including successful earlier batches
when a later listing page fails. Removed detail pages are skipped; exhausted
failures are logged and reported alongside all pipeline counters in the result.
Public login/challenge pages are reported as failures. The adapter does not
authenticate or bypass challenges.

## Checks

Run the standard-library suite without contacting LLM providers:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m unittest discover -s tests
```

The configuration and health routers are exercised in an isolated FastAPI app.
The existing main application/agent bootstrap is unfinished; these tests do not
claim that the full application can start. Startup wiring loads the configuration
manager before serving requests once that separate bootstrap work is completed.
