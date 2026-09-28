"""Frontend-owned configuration and provider diagnostics."""

from string import Formatter

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config.settings import Settings
from src.exceptions.config import ConfigError, LLMError
from src.infrastructure.llm.llm import LLMConfigManager, LLMProvider
from src.prompts.templates import (
    COVER_LETTER_PROMPT,
    DEFAULT_IMPROVE_PROMPT_ID,
    IMPROVE_PROMPT_OPTIONS,
    LANGUAGE_NAMES,
    OUTREACH_MESSAGE_PROMPT,
)
from src.schemas.api import (
    ApiKeyProviderStatus,
    ApiKeyStatusResponse,
    ApiKeysUpdateRequest,
    ApiKeysUpdateResponse,
    FeaturePromptsRequest,
    FeaturePromptsResponse,
    LanguageConfigRequest,
    LanguageConfigResponse,
    LLMConfigResponse,
    PromptConfigRequest,
    PromptConfigResponse,
    PromptOption,
    ResetDatabaseRequest,
)
from src.schemas.llm import (
    PROVIDERS,
    FeatureConfig,
    LLMConfig,
    LLMConfigRequest,
    LLMHealth,
    LLMTestRequest,
)

router = APIRouter(prefix="/config", tags=["Configuration"])


async def config_error_handler(request: Request, error: ConfigError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": {"code": error.code, "field": error.field}},
    )


async def llm_error_handler(request: Request, error: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": {"code": error.code}})


async def config_validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {key: item[key] for key in ("loc", "type", "msg")}
                for item in error.errors()
            ]
        },
    )


def _mask_api_key(key: str) -> str:
    return "..." + key[-4:] if len(key) > 4 else "*" * len(key)


def _llm_response(config: LLMConfig | None) -> LLMConfigResponse | None:
    if config is None:
        return None
    key = config.api_key.get_secret_value() if config.api_key else ""
    return LLMConfigResponse(**config.model_dump(), api_key=_mask_api_key(key))


@router.get("/llm-api-key")
@inject
async def get_llm_config_endpoint(
    config_manager: FromDishka[LLMConfigManager],
) -> LLMConfigResponse | None:
    return _llm_response(config_manager.get())


@router.put("/llm-api-key")
@inject
async def update_llm_config(
    config_manager: FromDishka[LLMConfigManager],
    request: LLMConfigRequest,
) -> LLMConfigResponse:
    return _llm_response(await config_manager.update(request))


@router.post("/llm-test")
@inject
async def test_llm_connection(
    config_manager: FromDishka[LLMConfigManager],
    settings: FromDishka[Settings],
    llm: FromDishka[LLMProvider],
    request: LLMTestRequest | None = None,
) -> LLMHealth:
    if request is not None:
        llm = LLMProvider(settings.llm, config_manager.resolve(request))
    return await llm.check_llm_health(include_details=True)


@router.get("/features")
@inject
async def get_feature_config(
    config_manager: FromDishka[LLMConfigManager],
) -> FeatureConfig | None:
    return config_manager.get_features()


@router.put("/features")
@inject
async def update_feature_config(
    config_manager: FromDishka[LLMConfigManager],
    request: FeatureConfig,
) -> FeatureConfig:
    return await config_manager.update_features(request)


def _language_response(config_manager: LLMConfigManager) -> LanguageConfigResponse:
    legacy = config_manager.get_value("language", "en")
    return LanguageConfigResponse(
        ui_language=config_manager.get_value("ui_language", legacy),
        content_language=config_manager.get_value("content_language", legacy),
        supported_languages=list(LANGUAGE_NAMES),
    )


@router.get("/language")
@inject
async def get_language_config(
    config_manager: FromDishka[LLMConfigManager],
) -> LanguageConfigResponse:
    return _language_response(config_manager)


@router.put("/language")
@inject
async def update_language_config(
    config_manager: FromDishka[LLMConfigManager],
    request: LanguageConfigRequest,
) -> LanguageConfigResponse:
    values = request.model_dump(exclude_none=True)
    for field, language in values.items():
        if language not in LANGUAGE_NAMES:
            raise ConfigError("unsupported_language", field=field, status_code=422)
    await config_manager.update_values(values)
    return _language_response(config_manager)


def _prompt_response(config_manager: LLMConfigManager) -> PromptConfigResponse:
    return PromptConfigResponse(
        default_prompt_id=config_manager.get_value(
            "default_prompt_id", DEFAULT_IMPROVE_PROMPT_ID
        ),
        prompt_options=[PromptOption(**option) for option in IMPROVE_PROMPT_OPTIONS],
    )


@router.get("/prompts")
@inject
async def get_prompt_config(
    config_manager: FromDishka[LLMConfigManager],
) -> PromptConfigResponse:
    return _prompt_response(config_manager)


@router.put("/prompts")
@inject
async def update_prompt_config(
    config_manager: FromDishka[LLMConfigManager],
    request: PromptConfigRequest,
) -> PromptConfigResponse:
    if request.default_prompt_id is not None:
        if request.default_prompt_id not in {
            option["id"] for option in IMPROVE_PROMPT_OPTIONS
        }:
            raise ConfigError(
                "unsupported_prompt", field="default_prompt_id", status_code=422
            )
        await config_manager.update_values(
            {"default_prompt_id": request.default_prompt_id}
        )
    return _prompt_response(config_manager)


def validate_prompt_placeholders(template: str) -> None:
    required = {"job_description", "resume_data", "output_language"}
    try:
        fields = set()
        for _, field, spec, conversion in Formatter().parse(template):
            if field is None:
                continue
            if field not in required or spec or conversion:
                raise ValueError
            fields.add(field)
        if fields != required:
            raise ValueError
    except ValueError:
        raise ConfigError("invalid_prompt_placeholders", status_code=422) from None


def _feature_prompts_response(
    config_manager: LLMConfigManager,
) -> FeaturePromptsResponse:
    return FeaturePromptsResponse(
        cover_letter_prompt=config_manager.get_value("cover_letter_prompt", ""),
        outreach_message_prompt=config_manager.get_value("outreach_message_prompt", ""),
        cover_letter_default=COVER_LETTER_PROMPT,
        outreach_message_default=OUTREACH_MESSAGE_PROMPT,
    )


@router.get("/feature-prompts")
@inject
async def get_feature_prompts(
    config_manager: FromDishka[LLMConfigManager],
) -> FeaturePromptsResponse:
    return _feature_prompts_response(config_manager)


@router.put("/feature-prompts")
@inject
async def update_feature_prompts(
    config_manager: FromDishka[LLMConfigManager],
    request: FeaturePromptsRequest,
) -> FeaturePromptsResponse:
    values = {
        name: value.strip()
        for name, value in request.model_dump(exclude_none=True).items()
    }
    for field, template in values.items():
        if template:
            try:
                validate_prompt_placeholders(template)
            except ConfigError as error:
                error.field = field
                raise
    await config_manager.update_values(values)
    return _feature_prompts_response(config_manager)


@router.get("/api-keys")
@inject
async def get_api_keys_status(
    config_manager: FromDishka[LLMConfigManager],
) -> ApiKeyStatusResponse:
    keys = config_manager.get_api_keys()
    return ApiKeyStatusResponse(
        providers=[
            ApiKeyProviderStatus(
                provider=provider,
                configured=bool(keys.get(provider)),
                masked_key=_mask_api_key(keys[provider].get_secret_value())
                if keys.get(provider)
                else None,
            )
            for provider in PROVIDERS
        ]
    )


@router.post("/api-keys")
@inject
async def update_api_keys(
    config_manager: FromDishka[LLMConfigManager],
    request: ApiKeysUpdateRequest,
) -> ApiKeysUpdateResponse:
    updates = request.model_dump(exclude_unset=True, exclude_none=True)
    await config_manager.update_api_keys(updates)
    return ApiKeysUpdateResponse(
        message=f"Updated {len(updates)} API key(s)",
        updated_providers=list(updates),
    )


@router.delete("/api-keys")
@inject
async def delete_all_api_keys(
    config_manager: FromDishka[LLMConfigManager],
    confirm: str | None = None,
) -> dict[str, str]:
    if confirm != "CLEAR_ALL_KEYS":
        raise HTTPException(status_code=400, detail="Pass confirm=CLEAR_ALL_KEYS")
    await config_manager.clear_api_keys()
    return {"message": "All API keys have been cleared"}


@router.delete("/api-keys/{provider}")
@inject
async def delete_api_key(
    config_manager: FromDishka[LLMConfigManager],
    provider: str,
) -> dict[str, str]:
    await config_manager.update_api_keys({provider: None})
    return {"message": f"API key for {provider} has been removed"}


@router.post("/reset")
async def reset_database_endpoint(request: ResetDatabaseRequest) -> dict[str, str]:
    if request.confirm != "RESET_ALL_DATA":
        raise HTTPException(status_code=400, detail="Pass confirm=RESET_ALL_DATA")
    raise HTTPException(status_code=501, detail="Database reset is not implemented")
