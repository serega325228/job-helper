from uuid import UUID

from dishka.integrations.taskiq import FromDishka, inject

from src.infrastructure.taskiq.broker import broker
from src.services.vacancy_scraping import VacancyScrapingService


@broker.task
@inject(patch_module=True)
async def scrape_company(
    company_id: UUID,
    service: FromDishka[VacancyScrapingService],
) -> None:
    await service.scrape_company(company_id)
