import asyncio

from sqlalchemy import text

from src.config.settings import get_settings
from src.infrastructure.db.engine import Database
from src.infrastructure.models import Base


async def upgrade() -> None:
    database = Database(get_settings().database)
    try:
        async with database.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(
                text("""
                ALTER TABLE vacancies
                ADD COLUMN IF NOT EXISTS preview_id UUID REFERENCES vacancy_previews(id),
                ADD COLUMN IF NOT EXISTS batch_id UUID REFERENCES vacancy_batches(id),
                ADD COLUMN IF NOT EXISTS processing_status VARCHAR(32) NOT NULL DEFAULT 'completed',
                ADD COLUMN IF NOT EXISTS raw_document JSONB,
                ADD COLUMN IF NOT EXISTS scores JSONB,
                ADD COLUMN IF NOT EXISTS evaluation JSONB,
                ADD COLUMN IF NOT EXISTS normalized_company TEXT GENERATED ALWAYS AS
                    (lower(trim(regexp_replace(company_name, '\\s+', ' ', 'g')))) STORED,
                ADD COLUMN IF NOT EXISTS normalized_title TEXT GENERATED ALWAYS AS
                    (lower(trim(regexp_replace(title, '\\s+', ' ', 'g')))) STORED
            """)
            )
            for statement in (
                "CREATE INDEX IF NOT EXISTS ix_vacancies_batch_id ON vacancies (batch_id)",
                "CREATE INDEX IF NOT EXISTS ix_vacancies_recent_reposts ON vacancies (normalized_company, normalized_title, discovered_at)",
                "CREATE INDEX IF NOT EXISTS ix_vacancies_recovery ON vacancies (processing_status, updated_at)",
            ):
                await connection.execute(text(statement))
    finally:
        await database.dispose()


if __name__ == "__main__":
    asyncio.run(upgrade())
