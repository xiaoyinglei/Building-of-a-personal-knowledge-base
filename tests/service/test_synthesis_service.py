from __future__ import annotations

from dataclasses import dataclass, field

from rag.retrieval.authorization_service import AuthorizationService
from rag.retrieval.synthesis_service import SynthesisService
from rag.schema.core import Document, DocumentType
from rag.schema.query import EvidenceItem
from rag.schema.runtime import AccessPolicy, ExecutionLocation, RuntimeMode


@dataclass
class _MetadataRepo:
    documents: dict[object, Document] = field(default_factory=dict)

    def get_document(self, doc_id):
        return self.documents.get(doc_id)


def _document(
    *,
    doc_id: int,
    is_active: bool = True,
    index_ready: bool = True,
    access_policy: AccessPolicy | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        source_id=1,
        doc_type=DocumentType.REPORT,
        file_hash=f"hash-{doc_id}",
        version_group_id=doc_id,
        is_active=is_active,
        index_ready=index_ready,
        effective_access_policy=access_policy or AccessPolicy.default(),
        embedding_model_id="bge-m3",
    )


def test_synthesis_service_filters_inactive_and_disallowed_documents() -> None:
    service = SynthesisService(
        metadata_repo=_MetadataRepo(
            documents={
                1: _document(doc_id=1),
                2: _document(doc_id=2, is_active=False, index_ready=False),
                3: _document(
                    doc_id=3,
                    access_policy=AccessPolicy(
                        allowed_locations=frozenset({ExecutionLocation.CLOUD}),
                        allowed_runtimes=frozenset({RuntimeMode.FAST, RuntimeMode.DEEP}),
                    ),
                ),
            }
        )
    )

    filtered = service.filter_evidence(
        evidence=[
            EvidenceItem(
                chunk_id="chunk-1",
                doc_id="1",
                citation_anchor="Allowed",
                text="allowed evidence",
                score=0.9,
            ),
            EvidenceItem(
                chunk_id="chunk-2",
                doc_id="2",
                citation_anchor="Inactive",
                text="inactive evidence",
                score=0.8,
            ),
            EvidenceItem(
                chunk_id="chunk-3",
                doc_id="3",
                citation_anchor="Cloud only",
                text="disallowed evidence",
                score=0.7,
            ),
        ],
        access_policy=AccessPolicy(
            allowed_locations=frozenset({ExecutionLocation.LOCAL}),
            allowed_runtimes=frozenset({RuntimeMode.FAST, RuntimeMode.DEEP}),
        ),
    )

    assert [item.chunk_id for item in filtered] == ["chunk-1"]


@dataclass
class _AuthorizationResolver:
    def allowed_doc_ids_for_user(self, user_id: str):
        return {"1"} if user_id == "alice" else set()


def test_synthesis_service_applies_user_scoped_authorization_view() -> None:
    service = SynthesisService(
        metadata_repo=_MetadataRepo(
            documents={
                1: _document(doc_id=1),
                2: _document(doc_id=2),
            }
        ),
        authorization_service=AuthorizationService(resolver=_AuthorizationResolver()),
    )

    filtered = service.filter_evidence(
        evidence=[
            EvidenceItem(chunk_id="chunk-1", doc_id="1", citation_anchor="Allowed", text="allowed evidence", score=0.9),
            EvidenceItem(chunk_id="chunk-2", doc_id="2", citation_anchor="Denied", text="denied evidence", score=0.8),
        ],
        access_policy=AccessPolicy.default(),
        user_id="alice",
    )

    assert [item.chunk_id for item in filtered] == ["chunk-1"]


def test_synthesis_service_denies_missing_document_metadata_by_default() -> None:
    service = SynthesisService(metadata_repo=_MetadataRepo(documents={}))

    filtered = service.filter_evidence(
        evidence=[
            EvidenceItem(chunk_id="chunk-1", doc_id="99", citation_anchor="Missing", text="missing evidence", score=0.4)
        ],
        access_policy=AccessPolicy.default(),
    )

    assert filtered == []


def test_synthesis_service_denies_when_user_view_cannot_be_resolved() -> None:
    service = SynthesisService(
        metadata_repo=_MetadataRepo(documents={1: _document(doc_id=1)}),
        authorization_service=AuthorizationService(resolver=object()),
    )

    filtered = service.filter_evidence(
        evidence=[
            EvidenceItem(chunk_id="chunk-1", doc_id="1", citation_anchor="Allowed?", text="maybe", score=0.9),
        ],
        access_policy=AccessPolicy.default(),
        user_id="alice",
    )

    assert filtered == []
