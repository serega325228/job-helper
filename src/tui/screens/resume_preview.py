class ResumePreviewScreen(
    ModalScreen[Literal["approve", "reject"] | None]
):
    def __init__(
        self,
        vacancy: Vacancy,
        resume: AdaptedResume,
    ) -> None:
        super().__init__()
        self.vacancy = vacancy
        self.resume = resume

    def compose(self) -> ComposeResult:
        changes = "\n".join(
            f"- {change}"
            for change in self.resume.changes
        )

        warnings = "\n".join(
            f"- {warning}"
            for warning in self.resume.warnings
        )

        with Vertical(id="resume-dialog"):
            yield Label(
                f"Предпросмотр: {self.vacancy.title}",
                id="resume-title",
            )

            with Horizontal(id="resume-columns"):
                with ScrollableContainer(
                    id="resume-content",
                ):
                    yield Markdown(self.resume.content)

                with ScrollableContainer(
                    id="resume-analysis",
                ):
                    yield Markdown(
                        f"""
## Изменения

{changes}

## Предупреждения checker-агента

{warnings}
"""
                    )

            with Horizontal(id="resume-actions"):
                yield Button(
                    "Одобрить и экспортировать",
                    id="approve",
                    variant="success",
                )
                yield Button(
                    "Отклонить",
                    id="reject",
                    variant="error",
                )
                yield Button(
                    "Закрыть",
                    id="close-preview",
                )

    @on(Button.Pressed, "#approve")
    def approve_resume(self) -> None:
        self.dismiss("approve")

    @on(Button.Pressed, "#reject")
    def reject_resume(self) -> None:
        self.dismiss("reject")

    @on(Button.Pressed, "#close-preview")
    def close_preview(self) -> None:
        self.dismiss(None)
