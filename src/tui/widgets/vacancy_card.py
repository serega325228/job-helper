class VacancyCard(Static):
    """
    Карточка одной вакансии.

    Карточка ничего не знает о supervisor, LangGraph или БД.
    Она только отображает данные и отправляет сообщения наверх.
    """

    class DetailsRequested(Message):
        def __init__(self, vacancy: Vacancy) -> None:
            super().__init__()
            self.vacancy = vacancy

    class AdaptRequested(Message):
        def __init__(self, vacancy: Vacancy) -> None:
            super().__init__()
            self.vacancy = vacancy

    def __init__(self, vacancy: Vacancy) -> None:
        super().__init__()
        self.vacancy = vacancy

    def compose(self) -> ComposeResult:
        salary = self.vacancy.salary or "Зарплата не указана"

        matching = ", ".join(self.vacancy.matching_skills)
        missing = ", ".join(self.vacancy.missing_skills)

        yield Label(
            self.vacancy.title,
            classes="vacancy-title",
        )

        yield Label(
            self.vacancy.company,
            classes="vacancy-company",
        )

        yield Label(salary)

        yield Label(
            f"Совпадение: {self.vacancy.match_score}%",
            classes="match-score",
        )

        yield Label(
            f"[green]Подходит:[/] {matching}",
        )

        yield Label(
            f"[yellow]Не хватает:[/] {missing}",
        )

        with Horizontal(classes="card-actions"):
            yield Button(
                "Подробнее",
                id="details",
            )
            yield Button(
                "Адаптировать резюме",
                id="adapt",
                variant="primary",
            )

    @on(Button.Pressed, "#details")
    def request_details(self) -> None:
        self.post_message(
            self.DetailsRequested(self.vacancy)
        )

    @on(Button.Pressed, "#adapt")
    def request_adaptation(self) -> None:
        self.post_message(
            self.AdaptRequested(self.vacancy)
        )
