"""Interview preparation generation service."""

import json
from typing import Any

from infrastructure.llm.llm import LLMProvider
from prompts.templates import INTERVIEW_PREP_PROMPT, get_language_name
from schemas.interview import InterviewPrepData

_JOB_DESCRIPTION_PROMPT_CHAR_LIMIT = 12_000
_RESUME_DATA_PROMPT_CHAR_LIMIT = 30_000
_TRUNCATION_NOTICE = (
    "[Content truncated for prompt length. Use only the visible evidence; "
    "do not infer or invent omitted details.]"
)


class InterviewPrepService:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    @staticmethod
    def _truncate_text_for_prompt(value: str, max_chars: int) -> str:
        """Bound unstructured prompt input while making omissions explicit."""
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars].rstrip()}\n\n{_TRUNCATION_NOTICE}"

    @staticmethod
    def _truncate_json_value(
        value: Any,
        *,
        max_string_chars: int,
        max_list_items: int,
    ) -> Any:
        if isinstance(value, str):
            return InterviewPrepService._truncate_text_for_prompt(
                value, max_string_chars
            )
        if isinstance(value, list):
            truncated = [
                InterviewPrepService._truncate_json_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                )
                for item in value[:max_list_items]
            ]
            if len(value) > max_list_items:
                truncated.append(
                    {
                        "_prompt_truncation_notice": (
                            f"{len(value) - max_list_items} additional items omitted. "
                            "Do not infer omitted details."
                        )
                    }
                )
            return truncated
        if isinstance(value, dict):
            return {
                key: InterviewPrepService._truncate_json_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                )
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _serialize_resume_data_for_prompt(resume_data: dict[str, Any]) -> str:
        resume_json = json.dumps(resume_data, ensure_ascii=False)
        if len(resume_json) <= _RESUME_DATA_PROMPT_CHAR_LIMIT:
            return resume_json

        for max_string_chars, max_list_items in ((2_000, 30), (1_000, 20), (500, 10)):
            bounded = InterviewPrepService._truncate_json_value(
                resume_data,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
            )
            bounded_json = json.dumps(bounded, ensure_ascii=False)
            if len(bounded_json) <= _RESUME_DATA_PROMPT_CHAR_LIMIT:
                return bounded_json

        compact_snapshot = json.dumps(
            InterviewPrepService._truncate_json_value(
                resume_data, max_string_chars=250, max_list_items=5
            ),
            ensure_ascii=False,
        )
        return json.dumps(
            {
                "_prompt_truncation_notice": _TRUNCATION_NOTICE,
                "limited_resume_snapshot": InterviewPrepService._truncate_text_for_prompt(
                    compact_snapshot,
                    _RESUME_DATA_PROMPT_CHAR_LIMIT - 500,
                ),
            },
            ensure_ascii=False,
        )
    # think about changing dict to pydantic model
    async def generate_interview_prep(
        self,
        resume_data: dict[str, Any],
        job_description: str,
        language: str = "en",
    ) -> InterviewPrepData:
        """Generate structured interview preparation for a tailored resume."""
        prompt = INTERVIEW_PREP_PROMPT.format(
            job_description=self._truncate_text_for_prompt(
                job_description,
                _JOB_DESCRIPTION_PROMPT_CHAR_LIMIT,
            ),
            resume_data=self._serialize_resume_data_for_prompt(resume_data),
            output_language=get_language_name(language),
        )

        max_tokens = self._llm.get_safe_max_tokens(requested=8192)

        result = await self._llm.complete(
            prompt=prompt,
            system_prompt=(
                "You are a career interview coach. Output truthful, resume-grounded "
                "interview preparation as JSON only."
            ),
            max_tokens=max_tokens,
            schema=InterviewPrepData,
        )

        return InterviewPrepData.model_validate(result)
