class ConfigError(Exception):
    def __init__(
        self,
        code: str = "configuration_required",
        *,
        field: str | None = None,
        status_code: int = 409,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.field = field
        self.status_code = status_code


class LLMError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
