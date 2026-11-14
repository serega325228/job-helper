"""Small reviewed title bank; exact aliases only, not a semantic role classifier."""

import json
import re
from hashlib import sha256

from src.schemas.preference import PreferenceData, RoleTitle

CATALOG_VERSION = "1"
ROLE_TITLES = (
    RoleTitle(id="software-engineer", title="Software Engineer", aliases=("Software Developer", "Разработчик ПО")),
    RoleTitle(id="backend-engineer", title="Backend Engineer", aliases=("Backend Developer", "Back-end Developer", "Бэкенд-разработчик")),
    RoleTitle(id="frontend-engineer", title="Frontend Engineer", aliases=("Frontend Developer", "Front-end Developer", "Фронтенд-разработчик")),
    RoleTitle(id="fullstack-engineer", title="Full Stack Engineer", aliases=("Fullstack Developer", "Full-stack Developer")),
    RoleTitle(id="mobile-engineer", title="Mobile Engineer", aliases=("Mobile Developer",)),
    RoleTitle(id="data-engineer", title="Data Engineer", aliases=("Инженер данных",)),
    RoleTitle(id="data-scientist", title="Data Scientist"),
    RoleTitle(id="ml-engineer", title="Machine Learning Engineer", aliases=("ML Engineer",)),
    RoleTitle(id="devops-engineer", title="DevOps Engineer"),
    RoleTitle(id="sre", title="Site Reliability Engineer", aliases=("SRE",)),
    RoleTitle(id="qa-engineer", title="QA Engineer", aliases=("Quality Assurance Engineer", "Тестировщик")),
    RoleTitle(id="product-manager", title="Product Manager", aliases=("Продакт-менеджер",)),
    RoleTitle(id="product-designer", title="Product Designer"),
    RoleTitle(id="business-analyst", title="Business Analyst", aliases=("Бизнес-аналитик",)),
)


def normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def canonical_title(value: str) -> str:
    for role in ROLE_TITLES:
        if normalized(value) in {normalized(item) for item in (role.title, *role.aliases)}:
            return role.title
    # Do not erase technology/seniority qualifiers or guess an unknown role's family.
    return " ".join(value.split())


def canonical_preference(data: PreferenceData) -> PreferenceData:
    return data.model_copy(update={
        "general_title": canonical_title(data.general_title),
        "target_titles": list(dict.fromkeys(canonical_title(item) for item in data.target_titles)),
    })


def preference_values(data: PreferenceData) -> dict:
    """Only fields supported by existing storage, never arbitrary model extras."""
    return data.model_dump(include=set(PreferenceData.model_fields))


def preference_signature(data: PreferenceData) -> str:
    values = preference_values(canonical_preference(data))
    # Preserve prose too: it may contain constraints not representable by the
    # current schema. Without target_titles the label is the only role constraint.
    if values["target_titles"]:
        values.pop("general_title")
    else:
        # Collision suffixes are storage labels, not new role constraints.
        label = re.sub(r" \[[0-9a-f]{12}(?:-\d+)?\]$", "", values["general_title"])
        values["general_title"] = canonical_title(label)
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = sorted({normalized(item) for item in value})
        elif isinstance(value, str):
            values[key] = normalized(value)
    return sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reconcile_preferences(
    drafts: list[PreferenceData], existing: list[PreferenceData],
    *, occupied_titles: list[str] | None = None,
) -> list[PreferenceData]:
    """Reuse full-criteria matches, append distinct goals, respect unique title storage."""
    signatures = {preference_signature(item) for item in existing}
    titles = {normalized(item.general_title) for item in existing}
    titles.update(normalized(title) for title in occupied_titles or [])
    result = []
    for draft in drafts:
        item = canonical_preference(draft)
        signature = preference_signature(item)
        if signature in signatures:
            continue
        title = item.general_title
        if normalized(title) in titles:
            title = f"{title[:181]} [{signature[:12]}]"
            suffix = 2
            while normalized(title) in titles:
                title = f"{item.general_title[:170]} [{signature[:12]}-{suffix}]"
                suffix += 1
        item = item.model_copy(update={"general_title": title})
        result.append(item)
        titles.add(normalized(title))
        signatures.add(signature)
    return result
