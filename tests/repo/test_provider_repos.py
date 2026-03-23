from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from pkp.repo.models.ollama_provider_repo import OllamaProviderRepo
from pkp.repo.models.openai_provider_repo import OpenAIProviderRepo


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.response_calls: list[dict[str, object]] = []
        self.embedding_calls: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self._create_response)
        self.embeddings = SimpleNamespace(create=self._create_embedding)

    def _create_response(self, **kwargs: object) -> object:
        self.response_calls.append(kwargs)
        return SimpleNamespace(output_text="adapter response")

    def _create_embedding(self, **kwargs: object) -> object:
        self.embedding_calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2]),
                SimpleNamespace(embedding=[0.3, 0.4]),
            ]
        )


def test_openai_provider_repo_lazily_creates_client_and_uses_responses_api() -> None:
    captured: list[FakeOpenAIClient] = []

    def factory() -> FakeOpenAIClient:
        client = FakeOpenAIClient()
        captured.append(client)
        return client

    repo = OpenAIProviderRepo(client_factory=factory, model="gpt-test")

    assert captured == []

    result = repo.chat("Summarize this")

    assert result == "adapter response"
    assert len(captured) == 1
    assert captured[0].response_calls == [{"model": "gpt-test", "input": "Summarize this"}]


def test_openai_provider_repo_uses_embeddings_api_and_fallback_rerank() -> None:
    client = FakeOpenAIClient()
    repo = OpenAIProviderRepo(client=client, embedding_model="embed-test")

    embeddings = repo.embed(["alpha", "beta"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert client.embedding_calls == [{"model": "embed-test", "input": ["alpha", "beta"]}]
    assert repo.rerank("query", ["a", "b", "c"]) == [0, 1, 2]


def test_ollama_provider_repo_uses_official_http_endpoints_and_fallback_rerank() -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        json_payload = json.loads(request.read().decode("utf-8"))
        requests.append((request.url.path, json_payload))
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "ollama response"}},
            )
        if request.url.path == "/api/embed":
            return httpx.Response(
                200,
                json={"embeddings": [[0.5, 0.6], [0.7, 0.8]]},
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://ollama.test") as client:
        repo = OllamaProviderRepo(
            http_client=client,
            base_url="http://ollama.test",
            chat_model="llama-test",
            embedding_model="embed-test",
        )

        chat = repo.chat("Explain adapters")
        embeddings = repo.embed(["alpha", "beta"])

    assert chat == "ollama response"
    assert embeddings == [[0.5, 0.6], [0.7, 0.8]]
    assert requests == [
        (
            "/api/chat",
            {
                "model": "llama-test",
                "messages": [{"role": "user", "content": "Explain adapters"}],
                "stream": False,
            },
        ),
        (
            "/api/embed",
            {
                "model": "embed-test",
                "input": ["alpha", "beta"],
            },
        ),
    ]
    assert repo.rerank("query", ["a", "b"]) == [0, 1]
