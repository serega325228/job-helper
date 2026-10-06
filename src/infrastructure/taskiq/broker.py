from aio_pika.exchange import ExchangeType
from taskiq import TaskiqEvents
from taskiq.middlewares import SmartRetryMiddleware
from taskiq_aio_pika.broker import AioPikaBroker
from taskiq_aio_pika.exchange import Exchange
from taskiq_aio_pika.queue import Queue, QueueType

from src.config.settings import get_settings

settings = get_settings()

exchange = Exchange(
    name=settings.taskiq.exchange_name,
    type=ExchangeType(settings.taskiq.exchange_type),
    durable=True,
)

io_queue = Queue(
    name=f"{settings.taskiq.exchange_name}.{settings.taskiq.io_queue}",
    routing_key=settings.taskiq.io_queue,
    type=QueueType.CLASSIC,
    durable=True,
    declare=True,
    auto_delete=False,
)

laya_queue = Queue(
    name=f"{settings.taskiq.exchange_name}.{settings.taskiq.laya_queue}",
    routing_key=settings.taskiq.laya_queue,
    type=QueueType.CLASSIC,
    durable=True,
    declare=True,
    auto_delete=False,
)

brokers = {
    queue.routing_key: AioPikaBroker(
        settings.taskiq.rabbitmq_url,
        exchange=exchange,
        task_queues=[queue],
        delay_queue=Queue(name=f"{queue.name}.delay", durable=True),
    ).with_middlewares(
        SmartRetryMiddleware(
            default_retry_count=settings.taskiq.stage_attempts,
            default_retry_label=True,
            default_delay=settings.taskiq.retry_delay_seconds,
            use_jitter=True,
            use_delay_exponent=True,
        )
    )
    for queue in (io_queue, laya_queue)
}
io_broker = brokers[settings.taskiq.io_queue]
laya_broker = brokers[settings.taskiq.laya_queue]
broker = io_broker


@io_broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def start_io_worker(state) -> None:
    from dishka.integrations.taskiq import setup_dishka

    from src.di.container import create_container

    state.container = create_container()
    setup_dishka(state.container, io_broker)
    await laya_broker.startup()


@laya_broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def start_laya_worker(state) -> None:
    from dishka.integrations.taskiq import setup_dishka

    from src.di.container import create_container

    state.container = create_container()
    setup_dishka(state.container, laya_broker)
    await io_broker.startup()


@io_broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def stop_io_worker(state) -> None:
    await laya_broker.shutdown()
    await state.container.close()


@laya_broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def stop_laya_worker(state) -> None:
    await io_broker.shutdown()
    await state.container.close()
