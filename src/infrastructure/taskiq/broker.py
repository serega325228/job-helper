import math
import random

from aio_pika import DeliveryMode, Message
from aio_pika.exchange import ExchangeType
from taskiq import BrokerMessage, TaskiqEvents, TaskiqMessage
from taskiq.kicker import AsyncKicker
from taskiq.middlewares import SmartRetryMiddleware
from taskiq_aio_pika.broker import AioPikaBroker
from taskiq_aio_pika.exchange import Exchange
from taskiq_aio_pika.queue import Queue, QueueType

from src.config.settings import get_settings

settings = get_settings()


class StageRetryMiddleware(SmartRetryMiddleware):
    def make_delay(self, message: TaskiqMessage, retries: int) -> float:
        base = float(message.labels.get("delay", self.default_delay))
        delay = base * 2 ** (retries - 1) if self.use_delay_exponent else base
        return min(delay, self.max_delay_exponent) + (
            random.uniform(0, base) if self.use_jitter else 0
        )

    async def on_send(
        self, kicker: AsyncKicker, message: TaskiqMessage, delay: float
    ) -> None:
        await kicker.with_labels(_stage_delay=delay).kiq(
            *message.args, **message.kwargs
        )


class VacancyBroker(AioPikaBroker):
    async def _declare_queues(self, channel):
        queues = await super()._declare_queues(channel)
        if channel is self.read_channel:
            route = getattr(settings.taskiq, f"{settings.taskiq.worker_queue}_queue")
            return [
                (queue, arguments)
                for queue, arguments in queues
                if queue.name == f"{self._exchange.name}.{route}"
            ]
        return queues

    async def kick(self, message: BrokerMessage) -> None:
        delay = message.labels.get("_stage_delay")
        if delay is None:
            await super().kick(message)
            return
        if self.write_channel is None:
            raise RuntimeError("Broker has not started")
        route = message.labels.get("queue_name")
        if route not in {queue.routing_key for queue in self._task_queues}:
            raise ValueError("Unknown task queue")
        seconds = math.ceil(float(delay))
        queue = await self.write_channel.declare_queue(
            f"{self._exchange.name}.{route}.retry.{seconds}",
            durable=True,
            arguments={
                "x-queue-type": QueueType.CLASSIC.value,
                "x-message-ttl": seconds * 1000,
                "x-expires": (seconds + 60) * 1000,
                "x-dead-letter-exchange": self._exchange.name,
                "x-dead-letter-routing-key": route,
            },
        )
        await self.write_channel.default_exchange.publish(
            Message(message.message, delivery_mode=DeliveryMode.PERSISTENT),
            routing_key=queue.name,
        )


broker = VacancyBroker(
    settings.taskiq.rabbitmq_url,
    qos=1 if settings.taskiq.worker_queue == "laya" else 10,
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
    if settings.taskiq.worker_queue == "laya":
        from src.infrastructure.laya.laya_provider import LayaProvider

        await (await state.container.get(LayaProvider)).load()


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def stop_worker(state) -> None:
    await state.container.close()
