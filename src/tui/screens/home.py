from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Markdown

from job_agent.domain.resume import AdaptedResume
from job_agent.domain.vacancy import Vacancy
from job_agent.tui.screens.resume_preview import ResumePreviewScreen
from job_agent.tui.screens.vacancy_details import VacancyDetailsScreen
from job_agent.tui.widgets.vacancy_card import VacancyCard

if TYPE_CHECKING:
    from job_agent.tui.app import JobAgentApp


class HomeScreen(Screen):
    BINDINGS = [
        ("escape", "focus_prompt", "Поле запроса"),
    ]

    @property
    def job_app(self) -> JobAgentApp:
        """
        self.app имеет общий тип App.

        Через property получаем конкретный JobAgentApp,
        чтобы IDE знала о supervisor_service и других сервисах.
        """
        from job_agent.tui.app import JobAgentApp

        return cast(JobAgentApp, self.app)

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label(
                    "Supervisor",
                    id="sidebar-title",
                )

                yield Markdown(
                    """
Напиши запрос обычным языком.

Например:

> Найди junior-вакансии Go
""",
                    id="conversation",
                )

                with Horizontal(id="prompt-row"):
                    yield Input(
                        placeholder="Сообщение supervisor...",
                        id="prompt",
                    )

                    yield Button(
                        "Отправить",
                        id="send",
                        variant="primary",
                    )

            with Vertical(id="content"):
                yield Label(
                    "Вакансии",
                    id="content-title",
                )

                yield Label(
                    "Результаты поиска появятся здесь.",
                    id="status",
                )

                yield ScrollableContainer(
                    id="vacancies",
                )

        yield Footer()

    def on_mount(self) -> None:
        self.focus_prompt()

    def focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_focus_prompt(self) -> None:
        self.focus_prompt()

    @on(Button.Pressed, "#send")
    def handle_send_button(self) -> None:
        self.submit_prompt()

    @on(Input.Submitted, "#prompt")
    def handle_prompt_submitted(
        self,
        event: Input.Submitted,
    ) -> None:
        self.submit_prompt()

    def submit_prompt(self) -> None:
        prompt = self.query_one("#prompt", Input)
        message = prompt.value.strip()

        if not message:
            self.notify(
                "Введите сообщение.",
                severity="warning",
            )
            return

        prompt.value = ""

        self.process_supervisor_message(message)

    @work(exclusive=True, group="supervisor")
    async def process_supervisor_message(
        self,
        message: str,
    ) -> None:
        status = self.query_one("#status", Label)
        conversation = self.query_one(
            "#conversation",
            Markdown,
        )

        conversation.update(
            f"""
## Вы

{message}

## Supervisor

Обрабатываю запрос...
"""
        )

        status.update("Supervisor выполняет workflow...")

        try:
            result = (
                await self.job_app.supervisor_service.handle_message(
                    message
                )
            )
        except Exception as exc:
            status.update("Ошибка supervisor.")

            self.notify(
                str(exc),
                severity="error",
            )
            return

        conversation.update(
            f"""
## Вы

{message}

## Supervisor

{result.message}
"""
        )

        status.update(result.message)

        if result.type == "vacancies":
            await self.render_vacancies(result.vacancies)

    async def render_vacancies(
        self,
        vacancies: tuple[Vacancy, ...],
    ) -> None:
        container = self.query_one(
            "#vacancies",
            ScrollableContainer,
        )

        await container.remove_children()

        for vacancy in vacancies:
            await container.mount(
                VacancyCard(vacancy)
            )

    @on(VacancyCard.DetailsRequested)
    def show_vacancy_details(
        self,
        message: VacancyCard.DetailsRequested,
    ) -> None:
        self.app.push_screen(
            VacancyDetailsScreen(message.vacancy)
        )

    @on(VacancyCard.AdaptRequested)
    def handle_adapt_request(
        self,
        message: VacancyCard.AdaptRequested,
    ) -> None:
        self.generate_resume_preview(
            message.vacancy
        )

    @work(exclusive=True, group="resume")
    async def generate_resume_preview(
        self,
        vacancy: Vacancy,
    ) -> None:
        status = self.query_one("#status", Label)

        status.update(
            f"Адаптирую резюме под «{vacancy.title}»..."
        )

        try:
            resume = (
                await self.job_app.resume_service.adapt_resume(
                    vacancy
                )
            )
        except Exception as exc:
            status.update(
                "Не удалось адаптировать резюме."
            )

            self.notify(
                str(exc),
                severity="error",
            )
            return

        status.update("Резюме готово к проверке.")

        self.app.push_screen(
            ResumePreviewScreen(
                vacancy=vacancy,
                resume=resume,
            ),
            lambda decision: self.handle_resume_decision(
                vacancy=vacancy,
                resume=resume,
                decision=decision,
            ),
        )

    def handle_resume_decision(
        self,
        vacancy: Vacancy,
        resume: AdaptedResume,
        decision: Literal["approve", "reject"] | None,
    ) -> None:
        if decision == "approve":
            self.export_resume(resume)
            return

        if decision == "reject":
            self.query_one("#status", Label).update(
                f"Резюме для «{vacancy.title}» отклонено."
            )

    @work(exclusive=True, group="export")
    async def export_resume(
        self,
        resume: AdaptedResume,
    ) -> None:
        status = self.query_one("#status", Label)

        status.update("Экспортирую резюме...")

        try:
            path = (
                await self.job_app.export_service.export_resume(
                    resume
                )
            )
        except Exception as exc:
            status.update("Ошибка экспорта.")

            self.notify(
                str(exc),
                severity="error",
            )
            return

        status.update(f"Резюме сохранено: {path}")

        self.notify(
            f"Файл сохранён: {path}",
            title="Экспорт завершён",
        )
