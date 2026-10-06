from aio_pika.exchange import ExchangeType
from taskiq_aio_pika.broker import AioPikaBroker
from taskiq_aio_pika.exchange import Exchange
from taskiq_aio_pika.queue import Queue, QueueType

from config.settings import get_settings

settings = get_settings()

exchange = Exchange(
    name=settings.taskiq.exchange_name,
    type=ExchangeType(settings.taskiq.exchange_type),
    durable=True,
)

io_queue = Queue(
    name=settings.taskiq.exchange_name + settings.taskiq.io_queue,
    routing_key=settings.taskiq.io_queue,
    type=QueueType.CLASSIC,
    durable=True,
    declare=True,
    auto_delete=False,
)

laya_queue = Queue(
    name=settings.taskiq.exchange_name + settings.taskiq.laya_queue,
    routing_key=settings.taskiq.laya_queue,
    type=QueueType.CLASSIC,
    durable=True,
    declare=True,
    auto_delete=False,
)

broker = AioPikaBroker(
    settings.taskiq.rabbitmq_url,
    exchange=exchange,
    queues=[io_queue, laya_queue],
)
