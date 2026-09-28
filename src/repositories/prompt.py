from src.infrastructure.llm.llm import LLMConfigManager


class PromptRepository:
    def __init__(self, config_manager: LLMConfigManager):
        self._config_manager = config_manager

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

        custom = (self._config_manager.get_value(custom_key) or "").strip()
        if not custom:
            return default_template, False
        return custom, True
