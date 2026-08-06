import logging

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


class RetryableLlmError(Exception):
    """Временная ошибка LLM: timeout, rate limit, 5xx."""


def llm_retry():
    return retry(
        retry=retry_if_exception_type(
            (RetryableLlmError, ValidationError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(
            initial=1,
            max=10,
            jitter=1,
        ),
        reraise=True,
        before_sleep=lambda state: logger.warning(
            "LLM retry: attempt=%s error=%r",
            state.attempt_number,
            state.outcome.exception() if state.outcome else None,
        ),
    )
