from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.engine import _CoreRAG
from rag.ingest.ingest import (
    BatchIngestResult,
    DeletePipelineResult,
    DirectContentItem,
    IngestPipelineResult,
    IngestRequest,
    RebuildPipelineResult,
)
from rag.llm.assembly import (
    AssemblyDiagnostics,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityBundle,
    CapabilityCatalog,
    CapabilityRequirements,
)
from rag.query.query import QueryOptions, RAGQueryResult
from rag.schema._types.text import TokenizerContract
from rag.schema.graph import GraphEdge, GraphNode
from rag.storage import StorageBundle, StorageConfig
from rag.utils._contracts import VisualDescriptionRepo
from rag.utils._telemetry import TelemetryService


@dataclass(slots=True)
class RAGRuntime:
    storage: StorageConfig
    request: AssemblyRequest = field(default_factory=AssemblyRequest)
    assembly_service: CapabilityAssemblyService = field(default_factory=CapabilityAssemblyService, repr=False)
    telemetry_service: TelemetryService | None = None
    vlm_repo: VisualDescriptionRepo | None = None
    capability_bundle: CapabilityBundle = field(init=False, repr=False)
    core: _CoreRAG = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.capability_bundle = self.assembly_service.assemble_request(self.request)
        self.core = _CoreRAG(
            storage=self.storage,
            assembly_service=self.assembly_service,
            capability_bundle=self.capability_bundle,
            telemetry_service=self.telemetry_service,
            vlm_repo=self.vlm_repo,
        )

    @classmethod
    def from_request(
        cls,
        *,
        storage: StorageConfig,
        request: AssemblyRequest,
        assembly_service: CapabilityAssemblyService | None = None,
        telemetry_service: TelemetryService | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
    ) -> RAGRuntime:
        return cls(
            storage=storage,
            request=request,
            assembly_service=assembly_service or CapabilityAssemblyService(),
            telemetry_service=telemetry_service,
            vlm_repo=vlm_repo,
        )

    @classmethod
    def from_profile(
        cls,
        *,
        storage: StorageConfig,
        profile_id: str,
        requirements: CapabilityRequirements | None = None,
        assembly_service: CapabilityAssemblyService | None = None,
        telemetry_service: TelemetryService | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
    ) -> RAGRuntime:
        service = assembly_service or CapabilityAssemblyService()
        request = service.request_for_profile(
            profile_id,
            requirements=requirements,
        )
        return cls.from_request(
            storage=storage,
            request=request,
            assembly_service=service,
            telemetry_service=telemetry_service,
            vlm_repo=vlm_repo,
        )

    @property
    def diagnostics(self) -> AssemblyDiagnostics:
        return self.capability_bundle.diagnostics

    @property
    def catalog(self) -> CapabilityCatalog:
        return self.assembly_service.catalog_from_environment(config=self.request.config)

    @property
    def token_contract(self) -> TokenizerContract:
        return self.capability_bundle.token_contract

    @property
    def runtime_contract_payload(self) -> dict[str, str | int | bool]:
        return self.capability_bundle.runtime_contract_payload

    @property
    def selected_profile_id(self) -> str | None:
        return self.capability_bundle.selected_profile_id

    @property
    def stores(self) -> StorageBundle:
        return self.core.stores

    def diagnostics_payload(self) -> dict[str, object]:
        return {
            "status": self.diagnostics.status,
            "selected_profile_id": self.selected_profile_id,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                }
                for issue in self.diagnostics.issues
            ],
            "decisions": [
                {
                    "capability": decision.capability,
                    "source": decision.source,
                    "provider_kind": decision.provider_kind,
                    "provider_name": decision.provider_name,
                    "model_name": decision.model_name,
                    "location": decision.location,
                    "reason": decision.reason,
                    "selected": decision.selected,
                }
                for decision in self.diagnostics.decisions
            ],
            "runtime_contract": self.runtime_contract_payload,
        }

    def close(self) -> None:
        self.core.stores.close()

    def __enter__(self) -> RAGRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def insert(self, request: IngestRequest | None = None, /, **kwargs: Any) -> IngestPipelineResult:
        return self.core.insert(request, **kwargs)

    def insert_many(
        self,
        requests: list[IngestRequest],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        return self.core.insert_many(requests, continue_on_error=continue_on_error)

    def insert_content_list(
        self,
        items: list[DirectContentItem],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        return self.core.insert_content_list(items, continue_on_error=continue_on_error)

    def query(
        self,
        *args: Any,
        options: QueryOptions | None = None,
        **kwargs: Any,
    ) -> RAGQueryResult:
        return self.core.query(*args, options=options, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> DeletePipelineResult:
        return self.core.delete(*args, **kwargs)

    def rebuild(self, *args: Any, **kwargs: Any) -> RebuildPipelineResult:
        return self.core.rebuild(*args, **kwargs)

    def upsert_node(self, node: GraphNode, *, evidence_chunk_ids: list[str] | None = None) -> GraphNode:
        return self.core.upsert_node(node, evidence_chunk_ids=evidence_chunk_ids)

    def upsert_edge(self, edge: GraphEdge, *, candidate: bool = False) -> GraphEdge:
        return self.core.upsert_edge(edge, candidate=candidate)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.core.get_node(node_id)

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        return self.core.list_nodes(node_type=node_type)

    def delete_node(self, node_id: str) -> None:
        self.core.delete_node(node_id)

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        return self.core.get_edge(edge_id, include_candidates=include_candidates)

    def list_edges(self) -> list[GraphEdge]:
        return self.core.list_edges()

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None:
        self.core.delete_edge(edge_id, include_candidates=include_candidates)

    def insert_custom_kg(
        self,
        *,
        nodes: list[GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
    ) -> dict[str, int]:
        return self.core.insert_custom_kg(nodes=nodes, edges=edges)


__all__ = ["RAGRuntime"]
