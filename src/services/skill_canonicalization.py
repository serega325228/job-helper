import re
import unicodedata
from collections.abc import Iterable

SEPARATOR_PATTERN = re.compile(r"[-_/]+")
SPACE_PATTERN = re.compile(r"\s+")


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = SEPARATOR_PATTERN.sub(" ", normalized)
    return SPACE_PATTERN.sub(" ", normalized)


SKILL_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    # Programming languages
    "python": ("python3", "py", "питон", "пайтон", "питон 3"),
    "go": ("golang", "go lang", "голанг", "го ланг"),
    "javascript": (
        "js",
        "java script",
        "джаваскрипт",
        "джава скрипт",
        "яваскрипт",
    ),
    "typescript": ("ts", "type script", "тайпскрипт", "тайп скрипт"),
    "java": ("java se", "джава", "ява"),
    "c#": ("c sharp", "csharp", "си шарп", "сишарп"),
    "c++": ("cpp", "c plus plus", "си плюс плюс"),
    "kotlin": ("котлин",),
    "swift": ("свифт",),
    "rust": ("раст", "rustlang", "rust lang"),
    "ruby": ("руби",),
    "php": ("пхп", "пи эйч пи"),
    "scala": ("скала",),
    "dart": ("дарт",),
    "lua": ("луа",),
    "bash": ("bash scripting", "bash script", "баш", "баш скрипты"),
    "powershell": ("power shell", "pwsh", "повершелл", "пауэршелл"),
    "sql": ("structured query language", "эскуэль", "эс кью эль"),
    # Web and application frameworks
    "node.js": ("node", "nodejs", "node js", "нод", "ноде", "нод джс"),
    "react": ("reactjs", "react.js", "react js", "реакт", "реакт джс"),
    "vue.js": ("vue", "vuejs", "vue js", "вью", "вью джс"),
    "angular": ("angular 2", "angular2", "ангуляр"),
    "angularjs": ("angular.js", "angular js", "ангуляр джс"),
    "next.js": ("next", "nextjs", "next js", "некст", "некст джс"),
    "nuxt.js": ("nuxt", "nuxtjs", "nuxt js", "накст", "накст джс"),
    "nestjs": ("nest", "nest.js", "nest js", "нест", "нест джс"),
    "express.js": ("express", "expressjs", "express js", "экспресс джс"),
    "django": ("джанго",),
    "django rest framework": ("drf", "django rest", "джанго рест"),
    "fastapi": ("fast api", "фастапи", "фаст апи"),
    "flask": ("фласк",),
    "spring boot": ("springboot", "spring-boot", "спринг бут"),
    "asp.net": ("aspnet", "asp net", "асп нет"),
    ".net": ("dotnet", "dot net", "дотнет", "дот нет"),
    "ruby on rails": ("rails", "ror", "руби он рейлс", "рейлс"),
    "laravel": ("ларавел",),
    "flutter": ("флаттер",),
    "react native": ("react-native", "реакт натив"),
    "celery": ("селери",),
    # Databases, search, and storage
    "postgresql": (
        "postgres",
        "postgre sql",
        "pgsql",
        "постгрес",
        "постгрэ",
        "постгрескл",
    ),
    "mysql": ("my sql", "майскл", "май эс кью эль"),
    "mariadb": ("maria db", "мариядб", "мария дб"),
    "microsoft sql server": (
        "mssql",
        "ms sql",
        "sql server",
        "майкрософт sql server",
    ),
    "sqlite": ("sqlite3", "sqlite 3", "скулайт", "эс кью лайт"),
    "mongodb": ("mongo", "mongo db", "монго", "монгодб", "монго дб"),
    "redis": ("редис",),
    "clickhouse": ("click house", "кликхаус", "клик хаус"),
    "elasticsearch": (
        "elastic search",
        "elastic",
        "эластик",
        "эластиксерч",
        "эластик серч",
    ),
    "opensearch": ("open search", "опенсерч", "опен серч"),
    "cassandra": ("apache cassandra", "кассандра"),
    "oracle database": ("oracle db", "oracle", "оракл", "оракл дб"),
    "minio": ("min io", "минио"),
    "amazon s3": ("aws s3", "s3", "амазон s3"),
    # Infrastructure and delivery
    "kubernetes": ("k8s", "кубер", "кубернетес", "кубернетис"),
    "docker": ("докер",),
    "docker compose": ("docker-compose", "dockercompose", "докер компоуз"),
    "terraform": ("терраформ",),
    "ansible": ("ансибл", "энсибл"),
    "helm": ("хелм", "хельм"),
    "nginx": ("энджинкс", "энжинкс", "нгинкс"),
    "linux": ("линукс", "gnu linux"),
    "git": ("гит",),
    "github actions": ("github-actions", "github action", "гитхаб экшнс"),
    "gitlab ci": ("gitlab-ci", "gitlab cicd", "гитлаб ci"),
    "jenkins": ("дженкинс",),
    "ci cd": ("ci/cd", "cicd", "си ай си ди"),
    "prometheus": ("прометеус", "прометей"),
    "grafana": ("графана",),
    "opentelemetry": ("open telemetry", "otel", "опентелеметри"),
    "elk": ("elk stack", "elastic stack", "елк", "елк стек"),
    # Cloud platforms
    "aws": ("amazon web services", "амазон веб сервисы", "авс"),
    "google cloud": ("gcp", "google cloud platform", "гугл клауд"),
    "microsoft azure": ("azure", "ms azure", "майкрософт ажур", "азур"),
    "yandex cloud": ("yc", "яндекс облако", "яндекс клауд"),
    "vk cloud": ("вк облако", "вк клауд"),
    # Messaging and distributed systems
    "apache kafka": ("kafka", "кафка", "апач кафка"),
    "rabbitmq": ("rabbit mq", "рэббит", "рэббитмк", "рэббит мк"),
    "nats": ("nats.io", "натс"),
    "activemq": ("active mq", "активмк", "актив мк"),
    "grpc": ("g rpc", "джи ар пи си"),
    "graphql": ("graph ql", "графкл", "граф кью эль"),
    "websocket": ("web socket", "websockets", "вебсокет", "веб сокеты"),
    "rest": ("rest api", "restful", "рест", "рест апи"),
    # Data, ML, and AI
    "pytorch": ("torch", "py torch", "пайторч"),
    "tensorflow": ("tensor flow", "тензорфлоу", "тензор флоу"),
    "scikit learn": ("scikit-learn", "sklearn", "сайкит лерн"),
    "pandas": ("пандас",),
    "numpy": ("num py", "нампай", "нумпай"),
    "large language models": ("llm", "llms", "большие языковые модели", "бям"),
    "natural language processing": ("nlp", "обработка естественного языка", "оея"),
    "computer vision": ("cv", "компьютерное зрение", "техническое зрение"),
    "machine learning": ("ml", "машинное обучение", "машобучение"),
    "deep learning": ("dl", "глубокое обучение"),
    "retrieval augmented generation": (
        "rag",
        "раг",
        "генерация с дополненным поиском",
    ),
    "langchain": ("lang chain", "лангчейн", "ланг чейн"),
    "hugging face": ("huggingface", "хаггинг фейс"),
    # Testing and engineering practices
    "pytest": ("py.test", "пайтест"),
    "junit": ("j unit", "джейюнит"),
    "selenium": ("селениум",),
    "playwright": ("плейрайт",),
    "microservices": (
        "microservice architecture",
        "микросервисы",
        "микросервисная архитектура",
    ),
    "domain driven design": (
        "ddd",
        "domain-driven design",
        "предметно ориентированное проектирование",
    ),
    "event driven architecture": (
        "eda",
        "event-driven architecture",
        "событийная архитектура",
    ),
}


def _build_aliases(
    groups: dict[str, Iterable[str]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, values in groups.items():
        normalized_canonical = _normalize(canonical)
        aliases[normalized_canonical] = normalized_canonical
        aliases.update(
            {
                _normalize(alias): normalized_canonical
                for alias in values
                if _normalize(alias)
            },
        )
    return aliases


DEFAULT_SKILL_ALIASES = _build_aliases(SKILL_ALIAS_GROUPS)


class SkillCanonicalizer:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._aliases = DEFAULT_SKILL_ALIASES.copy()
        self._aliases.update(
            {
                _normalize(alias): _normalize(canonical)
                for alias, canonical in (aliases or {}).items()
                if _normalize(alias) and _normalize(canonical)
            },
        )

    def canonicalize(self, value: str) -> str:
        normalized = _normalize(value)
        return self._aliases.get(normalized, normalized)

    def canonicalize_many(self, values: list[str]) -> set[str]:
        return {
            canonical for value in values if (canonical := self.canonicalize(value))
        }
