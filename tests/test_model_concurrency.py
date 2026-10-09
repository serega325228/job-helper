import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from unittest.mock import Mock, patch

from src.infrastructure.embedder.local import EmbeddingService
from src.infrastructure.laya.laya_provider import LayaProvider


class ModelConcurrencyTest(unittest.TestCase):
    def assert_serialized(self, methods, first, second):
        entered, release, second_started, overlap = (Event() for _ in range(4))
        state_lock = Lock()
        active = maximum = 0

        def inference(*args, **kwargs):
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
                if active > 1:
                    overlap.set()
            entered.set()
            try:
                self.assertTrue(release.wait(2), "Inference was not released")
                return [[0.5]]
            finally:
                with state_lock:
                    active -= 1

        for method in methods:
            method.side_effect = inference

        def start_second():
            second_started.set()
            return second()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_call = executor.submit(first)
            try:
                self.assertTrue(entered.wait(1))
                second_call = executor.submit(start_second)
                self.assertTrue(second_started.wait(1))
                self.assertFalse(overlap.wait(0.1), "Native model calls overlapped")
            finally:
                release.set()
            first_call.result(timeout=1)
            second_call.result(timeout=1)
        self.assertEqual(maximum, 1)

    def test_embedding_calls_share_one_guard(self):
        model = Mock()
        with patch("src.infrastructure.embedder.local.Llama", return_value=model):
            service = EmbeddingService("mock.gguf")
        self.assert_serialized(
            (model.embed,),
            lambda: asyncio.run(service.embed(["Python"])),
            lambda: asyncio.run(service.embed(["Developer\nPython"])),
        )
        self.assertEqual(model.embed.call_count, 2)

    def test_embedding_close_waits_for_inference_and_blocks_later_calls(self):
        model = Mock()
        entered, release, closing, closed = (Event() for _ in range(4))

        def inference(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            self.assertFalse(closed.is_set())
            return [[0.5]]

        model.embed.side_effect = inference
        model.close.side_effect = closed.set
        with patch("src.infrastructure.embedder.local.Llama", return_value=model):
            service = EmbeddingService("mock.gguf")

        def close():
            closing.set()
            service.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            call = executor.submit(asyncio.run, service.embed(["Python"]))
            try:
                self.assertTrue(entered.wait(1))
                cleanup = executor.submit(close)
                self.assertTrue(closing.wait(1))
                self.assertFalse(closed.wait(0.1))
            finally:
                release.set()
            call.result(timeout=1)
            cleanup.result(timeout=1)
        self.assertTrue(closed.is_set())
        service.close()
        model.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "Embedding service is closed"):
            asyncio.run(service.embed(["Python"]))
        model.embed.assert_called_once()


class LayaConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_laya_calls_and_cancellation_share_one_guard(self):
        model = Mock()
        model.predict.return_value = {"answers": {}}
        model.predict_batch.return_value = [{"answers": {}}]
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def offload(predict):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            return predict()

        with (
            patch(
                "src.infrastructure.laya.laya_provider.laya.load", return_value=model
            ) as load,
            patch(
                "src.infrastructure.laya.laya_provider.asyncio.to_thread",
                side_effect=offload,
            ),
        ):
            provider = LayaProvider()
            first = asyncio.create_task(provider.evaluate({"title": "AI"}, {}))
            await entered.wait()
            second = asyncio.create_task(provider.evaluate_batch([{"title": "AI"}], {}))
            first.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(calls, 1)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(await second, [{"answers": {}}])
            load.assert_called_once()
            model.predict.assert_called_once()
            model.predict_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
