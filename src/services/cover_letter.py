"""Cover letter, outreach message, and resume title generation service."""

import json
import logging
from typing import Any

from infrastructure.llm.llm import LLMProvider
from repositories.prompt import PromptRepository

from src.prompts.templates import (
    COVER_LETTER_PROMPT,
    GENERATE_TITLE_PROMPT,
    OUTREACH_MESSAGE_PROMPT,
    get_language_name,
)


class CoverLetterService:
    def __init__(
        self,
        llm: LLMProvider,
        prompts: PromptRepository
    ):
        self._llm = llm
        self._prompts = prompts


    async def generate_cover_letter(
        self,
        resume_data: dict[str, Any],
        job_description: str,
        language: str = "en",
    ) -> str:
        """Generate a cover letter based on resume and job description.

        Args:
            resume_data: Structured resume data (ResumeData format)
            job_description: Target job description text
            language: Output language code (en, es, zh, ja)

        Returns:
            Generated cover letter as plain text
        """
        output_language = get_language_name(language)

        template, is_custom = self._prompts.get(
            "cover_letter_prompt", COVER_LETTER_PROMPT
        )
        try:
            prompt = template.format(
                job_description=job_description,
                resume_data=json.dumps(resume_data),
                output_language=output_language,
            )
        except (KeyError, IndexError, ValueError) as e:
            # str.format() raises KeyError for unknown placeholders, IndexError for
            # positional out-of-range, and ValueError for unmatched/invalid braces
            # (e.g., ``{foo``). If the failing template is the built-in default,
            # something is broken upstream and the caller should see it — re-raise.
            # If it's a user-supplied custom prompt, fall back to the default with a
            # warning so generation doesn't crash on out-of-band disk edits.
            if not is_custom:
                raise
            logging.warning(
                "Custom cover letter prompt failed to format (%s); falling back to default",
                e,
            )
            prompt = COVER_LETTER_PROMPT.format(
                job_description=job_description,
                resume_data=json.dumps(resume_data),
                output_language=output_language,
            )

        result = await self._llm.complete(
            prompt=prompt,
            system_prompt="You are a professional career coach and resume writer. Write compelling, personalized cover letters.",
            max_tokens=2048,
        )

        return result.strip()


    async def generate_outreach_message(
        self,
        resume_data: dict[str, Any],
        job_description: str,
        language: str = "en",
    ) -> str:
        """Generate a cold outreach message for networking.

        Args:
            resume_data: Structured resume data (ResumeData format)
            job_description: Target job description text
            language: Output language code (en, es, zh, ja)

        Returns:
            Generated outreach message as plain text
        """
        output_language = get_language_name(language)

        template, is_custom = self._prompts.get(
            "outreach_message_prompt", OUTREACH_MESSAGE_PROMPT
        )
        try:
            prompt = template.format(
                job_description=job_description,
                resume_data=json.dumps(resume_data),
                output_language=output_language,
            )
        except (KeyError, IndexError, ValueError) as e:
            # See generate_cover_letter for rationale on the exception set.
            if not is_custom:
                raise
            logging.warning(
                "Custom outreach message prompt failed to format (%s); falling back to default",
                e,
            )
            prompt = OUTREACH_MESSAGE_PROMPT.format(
                job_description=job_description,
                resume_data=json.dumps(resume_data),
                output_language=output_language,
            )

        result = await self._llm.complete(
            prompt=prompt,
            system_prompt="You are a professional networking coach. Write genuine, engaging cold outreach messages.",
            max_tokens=1024,
        )

        return result.strip()


    async def generate_resume_title(
        self,
        job_description: str,
        language: str = "en",
    ) -> str:
        """Generate a short descriptive title from a job description.

        Args:
            job_description: Target job description text
            language: Output language code (en, es, zh, ja)

        Returns:
            Generated title like "Senior Frontend Engineer @ Stripe"
        """
        output_language = get_language_name(language)

        prompt = GENERATE_TITLE_PROMPT.format(
            job_description=job_description,
            output_language=output_language,
        )

        result = await self._llm.complete(
            prompt=prompt,
            system_prompt="You extract job titles and company names from job descriptions.",
            max_tokens=60,
            temperature=0.3,
        )

        # Strip quotes and whitespace, truncate to 80 chars
        title = result.strip().strip("\"'")
        return title[:80]
