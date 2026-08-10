import re
import unicodedata

DEFAULT_SKILL_ALIASES = {
    "golang": "go",
    "js": "javascript",
    "k8s": "kubernetes",
    "node js": "node.js",
    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "ts": "typescript",
}

SEPARATOR_PATTERN = re.compile(r"[-_/]+")
SPACE_PATTERN = re.compile(r"\s+")


class SkillCanonicalizer:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._aliases = DEFAULT_SKILL_ALIASES | (aliases or {})

    def canonicalize(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold().strip()
        normalized = SEPARATOR_PATTERN.sub(" ", normalized)
        normalized = SPACE_PATTERN.sub(" ", normalized)
        return self._aliases.get(normalized, normalized)

    def canonicalize_many(self, values: list[str]) -> set[str]:
        return {
            canonical
            for value in values
            if (canonical := self.canonicalize(value))
        }
