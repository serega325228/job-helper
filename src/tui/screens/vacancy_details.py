class VacancyDetailsScreen(ModalScreen[None]):
    def __init__(self, vacancy: Vacancy) -> None:
        super().__init__()
        self.vacancy = vacancy

    def compose(self) -> ComposeResult:
        matching = "\n".join(
            f"- {skill}"
            for skill in self.vacancy.matching_skills
        )

        missing = "\n".join(
            f"- {skill}"
            for skill in self.vacancy.missing_skills
        )

        content = f"""
# {self.vacancy.title}

**Компания:** {self.vacancy.company}

**Зарплата:** {self.vacancy.salary or "Не указана"}

**Совпадение:** {self.vacancy.match_score}%

## Описание

{self.vacancy.description}

## Подходящие навыки

{matching}

## Недостающие навыки

{missing}
"""

        with Vertical(id="details-dialog"):
            yield Markdown(content)
            yield Button(
                "Закрыть",
                id="close-details",
                variant="primary",
            )

    @on(Button.Pressed, "#close-details")
    def close_screen(self) -> None:
        self.dismiss()
