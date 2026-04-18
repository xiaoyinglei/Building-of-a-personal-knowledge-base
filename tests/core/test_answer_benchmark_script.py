from __future__ import annotations

from pathlib import Path

from rag.benchmarks import default_benchmark_paths, ensure_benchmark_layout
from scripts import evaluate_answer_benchmark
from scripts.evaluate_answer_benchmark import _resolve_default_answer_index


def test_answer_benchmark_defaults_to_bge_milvus_line_for_medical_mini() -> None:
    paths = ensure_benchmark_layout(default_benchmark_paths("medical_retrieval"), tasks=("retrieval", "ingest"))

    storage_root, vector_backend, collection_prefix = _resolve_default_answer_index(
        dataset="medical_retrieval",
        variant="mini",
        storage_root=None,
        vector_backend=None,
        vector_collection_prefix=None,
        embedding_provider=None,
        embedding_model=None,
        paths=paths,
    )

    assert storage_root.name == "mini-milvus-bge-v2"
    assert vector_backend == "milvus"
    assert collection_prefix == "medical_retrieval_mini_bge_v2"


def test_answer_benchmark_defaults_to_qwen8b_milvus_line_when_embedding_matches() -> None:
    paths = ensure_benchmark_layout(default_benchmark_paths("medical_retrieval"), tasks=("retrieval", "ingest"))

    storage_root, vector_backend, collection_prefix = _resolve_default_answer_index(
        dataset="medical_retrieval",
        variant="mini",
        storage_root=None,
        vector_backend=None,
        vector_collection_prefix=None,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:8b",
        paths=paths,
    )

    assert storage_root.name == "mini-milvus-qwen8b-v1"
    assert vector_backend == "milvus"
    assert collection_prefix == "medical_retrieval_mini_qwen8b_v1"


def test_answer_benchmark_main_passes_local_hf_chat_overrides(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def close(self) -> None:
            return None

    class _FakeEvaluator:
        def __init__(self, **kwargs) -> None:
            captured["evaluator_kwargs"] = kwargs

        def evaluate(self, **kwargs):
            captured["evaluate_kwargs"] = kwargs
            run_dir = tmp_path / "runs" / "answer"
            run_dir.mkdir(parents=True, exist_ok=True)
            return {
                "run_id": "run-1",
                "run_dir": str(run_dir),
                "summary": {"query_count": 1},
            }

    def _fake_build_runtime_for_benchmark(**kwargs):
        captured["runtime_kwargs"] = kwargs
        return _FakeRuntime()

    monkeypatch.setattr(
        evaluate_answer_benchmark,
        "build_runtime_for_benchmark",
        _fake_build_runtime_for_benchmark,
    )
    monkeypatch.setattr(evaluate_answer_benchmark, "build_chat_judge", lambda **kwargs: object())
    monkeypatch.setattr(evaluate_answer_benchmark, "AnswerBenchmarkEvaluator", _FakeEvaluator)

    exit_code = evaluate_answer_benchmark.main(
        [
            "--dataset",
            "medical_retrieval",
            "--variant",
            "mini",
            "--profile",
            "local_full",
            "--chat-provider",
            "local-hf",
            "--chat-model",
            "Qwen3-14B-4bit",
            "--chat-model-path",
            "/models/Qwen3-14B-4bit",
            "--chat-backend",
            "mlx",
            "--judge-subset-size",
            "0",
        ]
    )

    assert exit_code == 0
    runtime_kwargs = captured["runtime_kwargs"]
    assert runtime_kwargs["chat_provider_kind"] == "local-hf"
    assert runtime_kwargs["chat_model"] == "Qwen3-14B-4bit"
    assert runtime_kwargs["chat_model_path"] == "/models/Qwen3-14B-4bit"
    assert runtime_kwargs["chat_backend"] == "mlx"
    payload = capsys.readouterr().out
    assert '"chat_provider": "local-hf"' in payload
    assert '"chat_model_path_override": "/models/Qwen3-14B-4bit"' in payload
