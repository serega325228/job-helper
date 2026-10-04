import json
import unittest
from unittest.mock import Mock, patch

import httpx
from pydantic import ValidationError

from src.config.settings import EmbeddingSettings, RerankerSettings, Settings
from src.di.container import create_container
from src.infrastructure.embedder.llama_server import LlamaServerEmbedder
from src.infrastructure.reranker.llama_server import LlamaServerReranker
from src.ports.embedder import EMBEDDING_DIMENSIONS, Embedder
from src.ports.reranker import Reranker


class LlamaServerEmbeddingTest(unittest.IsolatedAsyncioTestCase):
    async def service(self, handler, *, batch_size=16):
        client = httpx.AsyncClient(
            base_url="http://llama.example/",
            transport=httpx.MockTransport(handler),
        )
        self.addAsyncCleanup(client.aclose)
        return LlamaServerEmbedder(client, "qwen-embedding", batch_size=batch_size)

    async def test_batches_restore_input_order_and_preserve_full_vectors(self):
        requests = []

        def handle(request):
            self.assertEqual(request.url.path, "/v1/embeddings")
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "index": index,
                            "embedding": ([3.0, 4.0] if index == 0 else [4.0, 3.0])
                            + [0.0] * (EMBEDDING_DIMENSIONS - 2),
                        }
                        for index in reversed(range(len(payload["input"])))
                    ]
                },
            )

        service = await self.service(handle, batch_size=2)
        vectors = await service.embed(["Python", "Go", "Rust"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual([len(payload["input"]) for payload in requests], [2, 1])
        self.assertEqual(requests[0]["model"], "qwen-embedding")
        self.assertEqual(requests[0]["encoding_format"], "float")
        self.assertEqual(requests[0]["input"], ["Python", "Go"])
        self.assertEqual(
            [vector[:2] for vector in vectors], [[3.0, 4.0], [4.0, 3.0], [3.0, 4.0]]
        )
        for vector in vectors:
            self.assertEqual(len(vector), EMBEDDING_DIMENSIONS)

    async def test_input_texts_are_sent_unchanged(self):
        prompts = []

        def handle(request):
            batch = json.loads(request.content)["input"]
            prompts.extend(batch)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [1.0] * EMBEDDING_DIMENSIONS}
                        for index in range(len(batch))
                    ]
                },
            )

        service = await self.service(handle)
        vectors = await service.embed(["Python", "Developer\nGo", "Engineer\nRust"])
        self.assertEqual(prompts, ["Python", "Developer\nGo", "Engineer\nRust"])
        self.assertEqual(len(vectors), 3)

    async def test_empty_batches_do_not_call_server(self):
        handler = Mock()
        service = await self.service(handler)
        self.assertEqual(await service.embed([]), [])
        handler.assert_not_called()

    async def test_invalid_responses_are_rejected(self):
        valid = {"index": 0, "embedding": [1.0] * EMBEDDING_DIMENSIONS}
        for payload in (
            {},
            {"data": []},
            {"data": [valid, valid]},
            {"data": [{**valid, "index": 1}]},
            {"data": [{**valid, "index": False}]},
            {"data": [{**valid, "embedding": [1.0]}]},
            {"data": [{**valid, "embedding": [1.0] * (EMBEDDING_DIMENSIONS + 1)}]},
            {"data": [{**valid, "embedding": [0.0] * EMBEDDING_DIMENSIONS}]},
            {"data": [{**valid, "embedding": ["nan"] * EMBEDDING_DIMENSIONS}]},
            {"data": [{**valid, "embedding": [True] * EMBEDDING_DIMENSIONS}]},
            {"data": [{**valid, "embedding": [[1.0] * EMBEDDING_DIMENSIONS]}]},
        ):
            with self.subTest(payload_type=str(payload)[:80]):
                service = await self.service(
                    lambda request: httpx.Response(200, json=payload)
                )
                with self.assertRaises(ValueError):
                    await service.embed(["Python"])

    async def test_http_and_transport_errors_propagate(self):
        service = await self.service(lambda request: httpx.Response(503))
        with self.assertRaises(httpx.HTTPStatusError):
            await service.embed(["Python"])

        def timeout(request):
            raise httpx.ReadTimeout("Timed out", request=request)

        service = await self.service(timeout)
        with self.assertRaises(httpx.ReadTimeout):
            await service.embed(["Python"])


class LlamaServerRerankerTest(unittest.IsolatedAsyncioTestCase):
    async def test_batches_restore_document_order_without_rescaling_scores(self):
        requests = []

        def handle(request):
            self.assertEqual(request.url.path, "/v1/rerank")
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": index, "relevance_score": (index + 1) / 10}
                        for index in reversed(range(len(payload["documents"])))
                    ]
                },
            )

        async with httpx.AsyncClient(
            base_url="http://llama.example/", transport=httpx.MockTransport(handle)
        ) as client:
            reranker = LlamaServerReranker(client, "qwen-reranker", batch_size=2)
            scores = await reranker.score("Python", ["First", "Second", "Third"])
        self.assertEqual(scores, [0.1, 0.2, 0.1])
        self.assertEqual([payload["top_n"] for payload in requests], [2, 1])
        self.assertEqual(requests[0]["query"], "Python")
        self.assertEqual(requests[0]["model"], "qwen-reranker")
        self.assertEqual(requests[1]["documents"], ["Third"])

    async def test_empty_documents_do_not_call_server(self):
        handler = Mock()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = LlamaServerReranker(client, "qwen-reranker")
            self.assertEqual(await reranker.score("Python", []), [])
        handler.assert_not_called()

    async def test_invalid_responses_are_rejected(self):
        valid = {"index": 0, "relevance_score": 0.8}
        for payload in (
            {},
            {"results": []},
            {"results": [valid, valid]},
            {"results": [{**valid, "index": 1}]},
            {"results": [{**valid, "index": False}]},
            {"results": [{**valid, "relevance_score": -0.1}]},
            {"results": [{**valid, "relevance_score": 1.1}]},
            {"results": [{**valid, "relevance_score": "nan"}]},
            {"results": [{**valid, "relevance_score": True}]},
        ):
            with self.subTest(payload=payload):
                async with httpx.AsyncClient(
                    base_url="http://llama.example/",
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, json=payload)
                    ),
                ) as client:
                    reranker = LlamaServerReranker(client, "qwen-reranker")
                    with self.assertRaises(ValueError):
                        await reranker.score("Python", ["Document"])

    async def test_http_and_transport_errors_propagate(self):
        for error in (httpx.ReadTimeout("Timed out"), None):
            def handle(request):
                if error:
                    raise error
                return httpx.Response(503)

            async with httpx.AsyncClient(
                base_url="http://llama.example/", transport=httpx.MockTransport(handle)
            ) as client:
                reranker = LlamaServerReranker(client, "qwen-reranker")
                with self.assertRaises(httpx.HTTPError):
                    await reranker.score("Python", ["Document"])


class ModelBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_selection_and_client_cleanup(self):
        settings = Settings(
            _env_file=None,
            embedding=EmbeddingSettings(
                _env_file=None, backend="llama_server", api_key="embedding-key"
            ),
            reranker=RerankerSettings(
                _env_file=None, backend="llama_server", api_key="reranker-key"
            ),
        )
        with patch("src.di.providers.get_settings", return_value=settings):
            async with create_container() as container:
                embedder = await container.get(Embedder)
                reranker = await container.get(Reranker)
                self.assertIsInstance(embedder, LlamaServerEmbedder)
                self.assertIsInstance(reranker, LlamaServerReranker)
                self.assertEqual(
                    embedder._client.headers["Authorization"], "Bearer embedding-key"
                )
                self.assertEqual(
                    reranker._client.headers["Authorization"], "Bearer reranker-key"
                )
                self.assertEqual(embedder._client.timeout.read, 120.0)
                self.assertEqual(reranker._client.timeout.read, 120.0)
                self.assertIs(await container.get(Embedder), embedder)
            self.assertTrue(embedder._client.is_closed)
            self.assertTrue(reranker._client.is_closed)

    async def test_local_selection_preserves_models_and_cleanup(self):
        settings = Settings(
            _env_file=None,
            embedding=EmbeddingSettings(_env_file=None, backend="local"),
            reranker=RerankerSettings(_env_file=None, backend="local"),
        )
        with (
            patch("src.di.providers.get_settings", return_value=settings),
            patch.object(EmbeddingSettings, "resolved_model_path", "mock.gguf"),
            patch("src.infrastructure.embedder.local.Llama") as llama,
            patch(
                "src.infrastructure.reranker.local.CrossEncoder"
            ) as encoder,
        ):
            async with create_container() as container:
                embedder = await container.get(Embedder)
                reranker = await container.get(Reranker)
                self.assertEqual(embedder.model_name, "mock")
                self.assertEqual(reranker.model_name, "BAAI/bge-reranker-v2-m3")
                llama.assert_called_once()
                encoder.assert_called_once()
            llama.return_value.close.assert_called_once()

    def test_invalid_settings_and_batch_sizes_are_rejected(self):
        for settings_type in (EmbeddingSettings, RerankerSettings):
            for values in (
                {"backend": "invalid"},
                {"base_url": "file:///model"},
                {"request_timeout_seconds": 0},
                {"batch_size": 0},
            ):
                with self.subTest(settings_type=settings_type, values=values):
                    with self.assertRaises(ValidationError):
                        settings_type(_env_file=None, **values)
        for adapter in (LlamaServerEmbedder, LlamaServerReranker):
            with self.assertRaises(ValueError):
                adapter(Mock(), "qwen", batch_size=0)


if __name__ == "__main__":
    unittest.main()
