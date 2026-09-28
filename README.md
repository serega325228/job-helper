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

## Checks

Run the standard-library suite without contacting LLM providers:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m unittest discover -s tests
```

The configuration and health routers are exercised in an isolated FastAPI app.
The existing main application/agent bootstrap is unfinished; these tests do not
claim that the full application can start. Startup wiring loads the configuration
manager before serving requests once that separate bootstrap work is completed.
