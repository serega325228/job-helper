from pathlib import Path
from config.settings import LLMSettings
import yaml


class PromptRepository:
    def __init__(self, settings: LLMSettings):
        self.path = settings.prompts_path
        self._prompts = self._load()

    def _load(self) -> dict[str, str]:
        with self.path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(
        self,
        custom_key: str,
        default_template: str,
    ) -> tuple[str, bool]:
        """
        Resolve a feature-prompt template at runtime.

        Returns ``(template, is_custom)``. If the stored custom prompt is
        empty or absent, returns the default template. The ``is_custom`` flag
        lets callers decide whether to fall back to the default on a format
        failure (defensive — save-time validation should have caught a
        malformed custom prompt).
        """

        custom = (self._prompts.get(custom_key) or "").strip()
        if not custom:
            return default_template, False
        return custom, True
