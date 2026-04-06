from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.query.context import CandidateLike
from rag.query.query import QueryMode
from rag.schema.chunk import ChunkRole
from rag.schema.document import AccessPolicy


@dataclass(frozen=True)
class FusedCandidate:
    candidate: CandidateLike
    fused_score: float
    rank: int
    supporting_branches: int
    branch_scores: dict[str, float]


@dataclass
class FusedCandidateView(CandidateLike):
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str
    source_id: str | None
    section_path: Sequence[str]
    effective_access_policy: AccessPolicy | None = None
    chunk_role: ChunkRole | None = None
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    parent_text: str | None = None
    metadata: dict[str, str] | None = None
    retrieval_channels: list[str] | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    special_score: float | None = None
    structure_score: float | None = None
    metadata_score: float | None = None
    fusion_score: float | None = None
    rrf_score: float | None = None
    unified_rank: int | None = None

    @property
    def item_id(self) -> str:
        return self.chunk_id


@dataclass(slots=True)
class ReciprocalRankFusion:
    rank_constant: int = 60

    def fuse(
        self,
        *,
        query: str,
        mode: QueryMode,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
    ) -> list[CandidateLike]:
        del query, mode
        fused: dict[str, FusedCandidate] = {}
        for branch_name, branch in branches:
            for index, candidate in enumerate(branch, start=1):
                score = 1.0 / (self.rank_constant + index)
                existing = fused.get(candidate.chunk_id)
                branch_scores = {branch_name: max(float(candidate.score), 0.0)}
                if existing is None:
                    fused[candidate.chunk_id] = FusedCandidate(
                        candidate=candidate,
                        fused_score=score,
                        rank=index,
                        supporting_branches=1,
                        branch_scores=branch_scores,
                    )
                    continue
                merged_scores = dict(existing.branch_scores)
                merged_scores.update(branch_scores)
                fused[candidate.chunk_id] = FusedCandidate(
                    candidate=existing.candidate,
                    fused_score=existing.fused_score + score,
                    rank=min(existing.rank, index),
                    supporting_branches=existing.supporting_branches + 1,
                    branch_scores=merged_scores,
                )

        ordered = sorted(
            fused.values(),
            key=lambda item: (-item.fused_score, -item.supporting_branches, item.rank, item.candidate.chunk_id),
        )
        return [self._to_view(item, index) for index, item in enumerate(ordered, start=1)]

    @staticmethod
    def _to_view(item: FusedCandidate, unified_rank: int) -> FusedCandidateView:
        final_score = item.fused_score
        return FusedCandidateView(
            chunk_id=item.candidate.chunk_id,
            doc_id=item.candidate.doc_id,
            text=item.candidate.text,
            citation_anchor=item.candidate.citation_anchor,
            score=final_score,
            rank=item.rank,
            source_kind=item.candidate.source_kind,
            source_id=item.candidate.source_id,
            section_path=tuple(item.candidate.section_path),
            effective_access_policy=getattr(item.candidate, "effective_access_policy", None),
            chunk_role=getattr(item.candidate, "chunk_role", None),
            special_chunk_type=getattr(item.candidate, "special_chunk_type", None),
            parent_chunk_id=getattr(item.candidate, "parent_chunk_id", None),
            parent_text=getattr(item.candidate, "parent_text", None),
            metadata=getattr(item.candidate, "metadata", None),
            retrieval_channels=sorted(item.branch_scores),
            dense_score=item.branch_scores.get("vector"),
            sparse_score=item.branch_scores.get("full_text"),
            special_score=item.branch_scores.get("special"),
            structure_score=item.branch_scores.get("section"),
            metadata_score=item.branch_scores.get("metadata"),
            fusion_score=final_score,
            rrf_score=final_score,
            unified_rank=unified_rank,
        )


__all__ = ["FusedCandidateView", "ReciprocalRankFusion"]
