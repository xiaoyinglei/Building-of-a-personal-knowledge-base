from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from rag.providers.adapters import (
    LocalHfChatProviderRepo,
    OllamaProviderRepo,
    OpenAIProviderRepo,
    _normalize_chat_text,
)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.response_calls: list[dict[str, object]] = []
        self.embedding_calls: list[dict[str, object]] = []
        self.chat_completion_calls: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self._create_response)
        self.embeddings = SimpleNamespace(create=self._create_embedding)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.fail_responses_with: Exception | None = None

    def _create_response(self, **kwargs: object) -> object:
        if self.fail_responses_with is not None:
            raise self.fail_responses_with
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

    def _create_chat_completion(self, **kwargs: object) -> object:
        self.chat_completion_calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="chat completion response"))])


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


def test_openai_provider_repo_uses_embeddings_api_and_rejects_rerank() -> None:
    client = FakeOpenAIClient()
    repo = OpenAIProviderRepo(client=client, embedding_model="embed-test")

    embeddings = repo.embed(["alpha", "beta"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert client.embedding_calls == [{"model": "embed-test", "input": ["alpha", "beta"]}]
    with pytest.raises(RuntimeError, match="does not provide rerank"):
        repo.rerank("query", ["a", "b", "c"])


def test_openai_provider_repo_falls_back_to_chat_completions_when_responses_404s() -> None:
    client = FakeOpenAIClient()
    client.fail_responses_with = RuntimeError("Error code: 404")
    repo = OpenAIProviderRepo(client=client, model="gpt-test")

    result = repo.chat("Summarize this")

    assert result == "chat completion response"
    assert client.response_calls == []
    assert client.chat_completion_calls == [
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "Summarize this"}],
        }
    ]


def test_openai_provider_repo_prefers_chat_completions_for_google_compatible_gateway() -> None:
    client = FakeOpenAIClient()
    repo = OpenAIProviderRepo(
        client=client,
        model="gemini-test",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )

    result = repo.chat("Reply with OK.")

    assert result == "chat completion response"
    assert client.response_calls == []
    assert client.chat_completion_calls == [
        {
            "model": "gemini-test",
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }
    ]


def test_ollama_provider_repo_uses_official_http_endpoints_and_rejects_rerank() -> None:
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
            batch_size=1,
        )

        repo.set_embedding_batch_size(2)
        repo.set_embedding_call_logging(True)
        repo.set_backend_progress_enabled(True)
        repo.set_device_preference("mps")

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
    assert repo.embedding_runtime_info() == {
        "provider": "ollama",
        "model_name": "embed-test",
        "device": "ollama-managed",
        "encode_batch_size": 2,
        "preferred_device": "mps",
    }
    stats = repo.embedding_stats()
    assert stats["provider"] == "ollama"
    assert stats["model_name"] == "embed-test"
    assert stats["device"] == "ollama-managed"
    assert stats["encode_batch_size"] == 2
    assert stats["preferred_device"] == "mps"
    assert stats["call_count"] == 1
    assert stats["text_count"] == 2
    assert stats["last_request_size"] == 2
    assert isinstance(stats["total_duration_ms"], float)
    assert isinstance(stats["last_duration_ms"], float)
    with pytest.raises(RuntimeError, match="does not provide rerank"):
        repo.rerank("query", ["a", "b"])


def test_local_hf_chat_provider_prefers_mlx_for_mlx_models(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalHfChatProviderRepo(chat_model="mlx-community/Qwen3-14B-4bit")

    monkeypatch.setattr(
        LocalHfChatProviderRepo,
        "_load_mlx_backend",
        lambda self: SimpleNamespace(chat=lambda prompt: f"mlx:{prompt}"),
    )
    monkeypatch.setattr(
        LocalHfChatProviderRepo,
        "_load_transformers_backend",
        lambda self: (_ for _ in ()).throw(AssertionError("transformers backend should not load first")),
    )

    assert provider.chat("hello") == "mlx:hello"
    assert provider.backend_name == "mlx"


def test_local_hf_chat_provider_falls_back_to_transformers_when_mlx_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalHfChatProviderRepo(chat_model="mlx-community/Qwen3-14B-4bit", backend="auto")

    monkeypatch.setattr(
        LocalHfChatProviderRepo,
        "_load_mlx_backend",
        lambda self: (_ for _ in ()).throw(RuntimeError("mlx unavailable")),
    )
    monkeypatch.setattr(
        LocalHfChatProviderRepo,
        "_load_transformers_backend",
        lambda self: SimpleNamespace(chat=lambda prompt: f"transformers:{prompt}"),
    )

    assert provider.chat("hello") == "transformers:hello"
    assert provider.backend_name == "transformers"


def test_normalize_chat_text_strips_qwen_think_blocks() -> None:
    assert _normalize_chat_text("<think>\ninternal\n</think>\n\nfinal answer") == "final answer"
    assert _normalize_chat_text("plain answer") == "plain answer"
