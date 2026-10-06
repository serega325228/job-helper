import asyncio

from aio_pika.exchange import ExchangeType
from taskiq import TaskiqEvents, TaskiqMessage
from taskiq.kicker import AsyncKicker
from taskiq.middlewares import SmartRetryMiddleware
from taskiq_aio_pika.broker import AioPikaBroker
from taskiq_aio_pika.exchange import Exchange
from taskiq_aio_pika.queue import Queue, QueueType

from src.config.settings import get_settings

settings = get_settings()


class StageRetryMiddleware(SmartRetryMiddleware):
    async def on_send(
        self, kicker: AsyncKicker, message: TaskiqMessage, delay: float
    ) -> None:
        # ponytail: backoff occupies a worker slot; use a delayed exchange if this limits throughput.
        await asyncio.sleep(delay)
        await kicker.kiq(*message.args, **message.kwargs)


broker = AioPikaBroker(
    settings.taskiq.rabbitmq_url,
    exchange=Exchange(
        name=settings.taskiq.exchange_name,
        type=ExchangeType(settings.taskiq.exchange_type),
        durable=True,
    ),
    task_queues=[
        Queue(
            name=f"{settings.taskiq.exchange_name}.{queue_name}",
            routing_key=queue_name,
            type=QueueType.CLASSIC,
            durable=True,
            declare=True,
            auto_delete=False,
        )
        for queue_name in (settings.taskiq.io_queue, settings.taskiq.laya_queue)
    ],
).with_middlewares(
    StageRetryMiddleware(
        default_retry_count=settings.taskiq.stage_attempts,
        default_retry_label=True,
        default_delay=settings.taskiq.retry_delay_seconds,
        use_jitter=True,
        use_delay_exponent=True,
    )
)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def start_worker(state) -> None:
    from dishka.integrations.taskiq import setup_dishka

    from src.di.container import create_container

    state.container = create_container()
    setup_dishka(state.container, broker)


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def stop_worker(state) -> None:
    await state.container.close()
