from __future__ import annotations

from textual.app import App

from job_agent.application.export_service import ExportService
from job_agent.application.resume_service import ResumeService
from job_agent.application.supervisor_service import SupervisorService
from job_agent.tui.screens.home import HomeScreen


class JobAgentApp(App):
    TITLE = "Job Agent"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+q", "quit", "Выход"),
        ("ctrl+l", "focus_prompt", "Запрос"),
    ]

    SCREENS = {
        "home": HomeScreen,
    }

    def __init__(
        self,
        supervisor_service: SupervisorService,
        resume_service: ResumeService,
        export_service: ExportService,
    ) -> None:
        super().__init__()

        self.supervisor_service = supervisor_service
        self.resume_service = resume_service
        self.export_service = export_service

    def on_mount(self) -> None:
        self.push_screen("home")

    def action_focus_prompt(self) -> None:
        home_screen = self.screen

        if isinstance(home_screen, HomeScreen):
            home_screen.focus_prompt()
