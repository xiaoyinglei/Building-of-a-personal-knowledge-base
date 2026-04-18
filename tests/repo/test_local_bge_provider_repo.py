from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from rag.providers.adapters import LocalBgeProviderRepo, _prepare_for_model_fallback


def test_local_bge_provider_repo_uses_snapshot_paths_for_embedding_and_rerank(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedding_root = tmp_path / "models--BAAI--bge-m3"
    embedding_snapshot = embedding_root / "snapshots" / "embedsha"
    embedding_snapshot.mkdir(parents=True)
    (embedding_root / "refs").mkdir(parents=True)
    (embedding_root / "refs" / "main").write_text("embedsha\n", encoding="utf-8")
    (embedding_snapshot / "config.json").write_text("{}", encoding="utf-8")

    rerank_root = tmp_path / "models--BAAI--bge-reranker-v2-m3"
    rerank_snapshot = rerank_root / "snapshots" / "reranksha"
    rerank_snapshot.mkdir(parents=True)
    (rerank_root / "refs").mkdir(parents=True)
    (rerank_root / "refs" / "main").write_text("reranksha\n", encoding="utf-8")
    (rerank_snapshot / "config.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeBGEM3FlagModel:
        def __init__(
            self,
            model_name_or_path: str,
            *,
            normalize_embeddings: bool,
            use_fp16: bool,
            devices: str | None,
            batch_size: int,
            query_max_length: int,
            passage_max_length: int,
            return_sparse: bool,
            return_colbert_vecs: bool,
        ) -> None:
            captured["embedding_model_name_or_path"] = model_name_or_path
            captured["normalize_embeddings"] = normalize_embeddings
            captured["embed_use_fp16"] = use_fp16
            captured["embed_devices"] = devices
            captured["embed_batch_size"] = batch_size
            captured["query_max_length"] = query_max_length
            captured["passage_max_length"] = passage_max_length
            captured["return_sparse"] = return_sparse
            captured["return_colbert_vecs"] = return_colbert_vecs
            self.target_devices = [devices or "cpu"]

        def encode(
            self,
            texts: list[str],
            *,
            batch_size: int,
            max_length: int,
            return_dense: bool,
            return_sparse: bool,
            return_colbert_vecs: bool,
        ) -> dict[str, list[list[float]]]:
            captured["embed_texts"] = texts
            captured["encode_batch_size"] = batch_size
            captured["encode_max_length"] = max_length
            captured["return_dense"] = return_dense
            captured["encode_return_sparse"] = return_sparse
            captured["encode_return_colbert_vecs"] = return_colbert_vecs
            return {"dense_vecs": [[0.1, 0.2], [0.3, 0.4]]}

    class FakeAutoReranker:
        @staticmethod
        def from_finetuned(
            model_name_or_path: str,
            *,
            model_class: str,
            use_fp16: bool,
            trust_remote_code: bool,
            devices: str | None,
            batch_size: int,
            max_length: int,
            normalize: bool,
        ) -> FakeAutoReranker:
            captured["rerank_model_name_or_path"] = model_name_or_path
            captured["rerank_model_class"] = model_class
            captured["rerank_use_fp16"] = use_fp16
            captured["rerank_trust_remote_code"] = trust_remote_code
            captured["rerank_devices"] = devices
            captured["rerank_batch_size"] = batch_size
            captured["rerank_max_length"] = max_length
            captured["rerank_normalize"] = normalize
            return FakeAutoReranker()

        def compute_score(
            self,
            pairs: list[list[str]],
            *,
            batch_size: int,
            max_length: int,
        ) -> list[float]:
            captured["pairs"] = pairs
            captured["compute_batch_size"] = batch_size
            captured["compute_max_length"] = max_length
            return [0.2, 0.9]

    monkeypatch.setattr(
        "rag.providers.adapters.importlib.import_module",
        lambda _name: SimpleNamespace(BGEM3FlagModel=FakeBGEM3FlagModel, FlagAutoReranker=FakeAutoReranker),
    )
    provider = LocalBgeProviderRepo(
        embedding_model="BAAI/bge-m3",
        embedding_model_path=str(embedding_root),
        rerank_model="BAAI/bge-reranker-v2-m3",
        rerank_model_path=str(rerank_root),
        batch_size=4,
        max_length=1024,
        rerank_batch_size=3,
        rerank_max_length=256,
    )

    embeddings = provider.embed(["alpha", "beta"])
    ranking = provider.rerank("query", ["candidate-a", "candidate-b"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert ranking == [1, 0]
    assert captured["embedding_model_name_or_path"] == str(embedding_snapshot)
    assert captured["rerank_model_name_or_path"] == str(rerank_snapshot)
    assert captured["rerank_model_class"] == "encoder-only-base"
    assert captured["embed_devices"] == provider.embedding_device
    assert captured["rerank_devices"] == provider.embedding_device
    assert captured["embed_texts"] == ["alpha", "beta"]
    assert captured["pairs"] == [["query", "candidate-a"], ["query", "candidate-b"]]

    object_ranking = provider.rerank(
        "query",
        [
            SimpleNamespace(text="candidate-a"),
            SimpleNamespace(text="candidate-b"),
        ],
    )

    assert object_ranking == [1, 0]
    assert captured["pairs"] == [["query", "candidate-a"], ["query", "candidate-b"]]


def test_local_bge_provider_repo_suppresses_fast_tokenizer_padding_warning(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedding_root = tmp_path / "models--BAAI--bge-m3"
    embedding_snapshot = embedding_root / "snapshots" / "embedsha"
    embedding_snapshot.mkdir(parents=True)
    (embedding_root / "refs").mkdir(parents=True)
    (embedding_root / "refs" / "main").write_text("embedsha\n", encoding="utf-8")
    (embedding_snapshot / "config.json").write_text("{}", encoding="utf-8")

    rerank_root = tmp_path / "models--BAAI--bge-reranker-v2-m3"
    rerank_snapshot = rerank_root / "snapshots" / "reranksha"
    rerank_snapshot.mkdir(parents=True)
    (rerank_root / "refs").mkdir(parents=True)
    (rerank_root / "refs" / "main").write_text("reranksha\n", encoding="utf-8")
    (rerank_snapshot / "config.json").write_text("{}", encoding="utf-8")

    class FakeFastTokenizer:
        is_fast = True

        def __init__(self) -> None:
            self.deprecation_warnings: dict[str, bool] = {}

        def pad(self, *_args: object, **_kwargs: object) -> None:
            if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
                warnings.warn(
                    "You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, "
                    "using the `__call__` method is faster than using a method to encode the text followed by a "
                    "call to the `pad` method to get a padded encoding.",
                    UserWarning,
                    stacklevel=2,
                )

    class FakeBGEM3FlagModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.tokenizer = FakeFastTokenizer()

        def encode(self, *_args: object, **_kwargs: object) -> dict[str, list[list[float]]]:
            self.tokenizer.pad([{"input_ids": [1, 2, 3]}], padding=True)
            return {"dense_vecs": [[0.1, 0.2]]}

    class FakeAutoReranker:
        @staticmethod
        def from_finetuned(*_args: object, **_kwargs: object) -> FakeAutoReranker:
            return FakeAutoReranker()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.tokenizer = FakeFastTokenizer()

        def compute_score(self, pairs: list[list[str]], **_kwargs: object) -> list[float]:
            self.tokenizer.pad([{"input_ids": [1, 2, 3]}], padding=True)
            return [0.8 for _ in pairs]

    monkeypatch.setattr(
        "rag.providers.adapters.importlib.import_module",
        lambda _name: SimpleNamespace(BGEM3FlagModel=FakeBGEM3FlagModel, FlagAutoReranker=FakeAutoReranker),
    )
    provider = LocalBgeProviderRepo(
        embedding_model_path=str(embedding_root),
        rerank_model_path=str(rerank_root),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider.embed(["alpha"])
        provider.rerank("query", ["candidate-a"])

    assert not any("using the `__call__` method is faster" in str(item.message) for item in caught)


def test_local_bge_provider_repo_reports_embedding_runtime_and_stats(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedding_root = tmp_path / "models--BAAI--bge-m3"
    embedding_snapshot = embedding_root / "snapshots" / "embedsha"
    embedding_snapshot.mkdir(parents=True)
    (embedding_root / "refs").mkdir(parents=True)
    (embedding_root / "refs" / "main").write_text("embedsha\n", encoding="utf-8")
    (embedding_snapshot / "config.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeBGEM3FlagModel:
        def __init__(
            self,
            *_args: object,
            devices: str | None,
            batch_size: int,
            **_kwargs: object,
        ) -> None:
            captured["devices"] = devices
            captured["batch_size"] = batch_size
            self.target_devices = [devices or "cpu"]

        def encode(self, texts: list[str], **_kwargs: object) -> dict[str, list[list[float]]]:
            return {"dense_vecs": [[0.1, 0.2] for _ in texts]}

    monkeypatch.setattr(
        "rag.providers.adapters.importlib.import_module",
        lambda _name: SimpleNamespace(BGEM3FlagModel=FakeBGEM3FlagModel, FlagAutoReranker=object),
    )

    provider = LocalBgeProviderRepo(
        embedding_model_path=str(embedding_root),
        batch_size=16,
        devices="mps",
    )
    provider.set_embedding_call_logging(True)
    provider.embed(["alpha", "beta", "gamma"])

    runtime_info = provider.embedding_runtime_info()
    stats = provider.embedding_stats()

    assert captured["devices"] == "mps"
    assert captured["batch_size"] == 16
    assert runtime_info["device"] == "mps"
    assert runtime_info["encode_batch_size"] == 16
    assert stats["call_count"] == 1
    assert stats["text_count"] == 3


def test_local_bge_provider_repo_uses_decoder_only_reranker_for_qwen_models(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    rerank_root = tmp_path / "models--Qwen--Qwen3-Reranker-4B"
    rerank_snapshot = rerank_root / "snapshots" / "reranksha"
    rerank_snapshot.mkdir(parents=True)
    (rerank_root / "refs").mkdir(parents=True)
    (rerank_root / "refs" / "main").write_text("reranksha\n", encoding="utf-8")
    (rerank_snapshot / "config.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeAutoReranker:
        @staticmethod
        def from_finetuned(model_name_or_path: str, **kwargs: object) -> FakeAutoReranker:
            captured["model_name_or_path"] = model_name_or_path
            captured.update(kwargs)
            return FakeAutoReranker()

        def compute_score(self, pairs: list[list[str]], **_kwargs: object) -> list[float]:
            return [0.9 for _ in pairs]

    monkeypatch.setattr(
        "rag.providers.adapters.importlib.import_module",
        lambda _name: SimpleNamespace(BGEM3FlagModel=object, FlagAutoReranker=FakeAutoReranker),
    )

    provider = LocalBgeProviderRepo(
        embedding_model="BAAI/bge-m3",
        rerank_model="Qwen/Qwen3-Reranker-4B",
        rerank_model_path=str(rerank_root),
    )
    provider.rerank("query", ["candidate-a"])

    assert captured["model_name_or_path"] == str(rerank_snapshot)
    assert captured["model_class"] == "decoder-only-base"
    assert captured["trust_remote_code"] is True


def test_prepare_for_model_fallback_handles_roberta_pair_sequences() -> None:
    class FakeXLMRobertaTokenizer:
        cls_token_id = 0
        sep_token_id = 2
        bos_token_id = None
        eos_token_id = None

        @staticmethod
        def num_special_tokens_to_add(*, pair: bool) -> int:
            return 4 if pair else 2

    tokenizer = FakeXLMRobertaTokenizer()

    encoded = _prepare_for_model_fallback(
        tokenizer,
        [11, 12],
        [21, 22, 23],
        truncation="only_second",
        max_length=8,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=True,
    )

    assert encoded["input_ids"] == [0, 11, 12, 2, 2, 21, 22, 2]


def test_prepare_for_model_fallback_handles_decoder_only_sequences_without_special_tokens() -> None:
    class FakeQwen2Tokenizer:
        cls_token_id = None
        sep_token_id = None
        bos_token_id = 151643
        eos_token_id = 151645

        @staticmethod
        def num_special_tokens_to_add(*, pair: bool) -> int:
            return 0

    tokenizer = FakeQwen2Tokenizer()

    encoded = _prepare_for_model_fallback(
        tokenizer,
        [1, 2, 3],
        [4, 5, 6, 7],
        truncation="only_second",
        max_length=5,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False,
    )

    assert encoded["input_ids"] == [1, 2, 3, 4, 5]
