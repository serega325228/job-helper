class VacancyNormalizationError(RuntimeError):
    """Raised when raw vacancies cannot be mapped to normalized vacancies."""


class VacancyScrapingError(RuntimeError):
    """Raised when a source page cannot be safely parsed."""


class VacancyPreviewEvaluationError(RuntimeError):
    """Raised when preview evaluation cannot be mapped to its input batch."""
