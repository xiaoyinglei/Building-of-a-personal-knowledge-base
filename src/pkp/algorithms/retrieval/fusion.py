from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pkp.query.query import QueryMode
from pkp.query.context import CandidateLike
from pkp.schema.chunk import ChunkRole
from pkp.schema.document import AccessPolicy
from pkp.types.text import (
    looks_command_like,
    looks_definition_query,
    looks_definition_text,
    looks_operation_query,
    looks_structure_query,
    looks_structure_text,
)


@dataclass(frozen=True)
class FusedCandidate:
    candidate: CandidateLike
    fused_score: float
    rank: int
    supporting_branches: int
    branch_scores: dict[str, float]


@dataclass
class FusedCandidateView:
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


@dataclass(slots=True)
class ReciprocalRankFusion:
    def fuse(
        self,
        *,
        query: str,
        mode: QueryMode,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
    ) -> list[CandidateLike]:
        fused: dict[str, FusedCandidate] = {}
        k = 60
        for branch_name, branch in branches:
            branch_weight = self._branch_weight(branch_name, query=query, mode=mode)
            normalized_scores = self._normalized_branch_scores(branch)
            for index, candidate in enumerate(branch, start=1):
                normalized_score = normalized_scores.get(candidate.chunk_id, 0.0)
                score = (
                    branch_weight / (k + index)
                    + (branch_weight * normalized_score * 0.3)
                    + self._candidate_quality_prior(query, candidate)
                )
                existing = fused.get(candidate.chunk_id)
                if existing is None:
                    fused[candidate.chunk_id] = FusedCandidate(
                        candidate=candidate,
                        fused_score=score,
                        rank=index,
                        supporting_branches=1,
                        branch_scores={branch_name: max(float(candidate.score), 0.0)},
                    )
                    continue
                branch_scores = dict(existing.branch_scores)
                branch_scores[branch_name] = max(float(candidate.score), 0.0)
                fused[candidate.chunk_id] = FusedCandidate(
                    candidate=existing.candidate,
                    fused_score=existing.fused_score + score,
                    rank=min(existing.rank, index),
                    supporting_branches=existing.supporting_branches + 1,
                    branch_scores=branch_scores,
                )

        ordered = sorted(
            fused.values(),
            key=lambda fused_candidate: (
                -(fused_candidate.fused_score + max(0, fused_candidate.supporting_branches - 1) * 0.05),
                -fused_candidate.supporting_branches,
                fused_candidate.rank,
            ),
        )
        return [
            FusedCandidateView(
                chunk_id=item.candidate.chunk_id,
                doc_id=item.candidate.doc_id,
                text=item.candidate.text,
                citation_anchor=item.candidate.citation_anchor,
                score=item.fused_score + max(0, item.supporting_branches - 1) * 0.05,
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
                fusion_score=item.fused_score + max(0, item.supporting_branches - 1) * 0.05,
                rrf_score=item.fused_score + max(0, item.supporting_branches - 1) * 0.05,
                unified_rank=index,
            )
            for index, item in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _normalized_branch_scores(branch: Sequence[CandidateLike]) -> dict[str, float]:
        positive_scores = [max(float(candidate.score), 0.0) for candidate in branch]
        max_score = max(positive_scores, default=0.0)
        if max_score > 0.0:
            return {
                candidate.chunk_id: max(float(candidate.score), 0.0) / max_score
                for candidate in branch
            }
        size = len(branch)
        if size <= 1:
            return {candidate.chunk_id: 1.0 for candidate in branch}
        return {
            candidate.chunk_id: 1.0 - ((index - 1) / size)
            for index, candidate in enumerate(branch, start=1)
        }

    @staticmethod
    def _branch_weight(branch_name: str, *, query: str, mode: QueryMode) -> float:
        if mode is QueryMode.NAIVE:
            return {"vector": 1.4}.get(branch_name, 0.0)
        if mode is QueryMode.LOCAL:
            return {"local": 1.35}.get(branch_name, 0.0)
        if mode is QueryMode.GLOBAL:
            return {"global": 1.3}.get(branch_name, 0.0)
        if mode is QueryMode.HYBRID:
            return {"local": 1.25, "global": 1.2}.get(branch_name, 0.0)

        if looks_structure_query(query):
            return {
                "full_text": 1.0,
                "vector": 0.95,
                "local": 1.15,
                "global": 1.1,
                "section": 1.3,
                "metadata": 1.2,
                "web": 0.6,
            }.get(branch_name, 1.0)
        if looks_definition_query(query):
            return {
                "full_text": 0.9,
                "vector": 1.2,
                "local": 1.25,
                "global": 1.05,
                "section": 0.8,
                "metadata": 0.9,
                "web": 0.6,
            }.get(branch_name, 1.0)
        if looks_operation_query(query):
            return {
                "full_text": 1.1,
                "vector": 1.0,
                "local": 1.15,
                "global": 1.05,
                "section": 0.9,
                "metadata": 0.9,
                "web": 0.6,
            }.get(branch_name, 1.0)
        return {
            "full_text": 1.0,
            "vector": 1.1,
            "local": 1.15,
            "global": 1.1,
            "section": 0.9,
            "metadata": 1.0,
            "web": 0.6,
        }.get(branch_name, 1.0)

    @staticmethod
    def _candidate_quality_prior(query: str, candidate: CandidateLike) -> float:
        text = candidate.text
        section_text = " ".join(candidate.section_path)
        query_is_command_like = looks_command_like(query)
        query_is_definition_like = looks_definition_query(query)
        query_is_structure_like = looks_structure_query(query)

        prior = 0.0
        if not query_is_command_like and looks_command_like(text):
            prior -= 0.35
            if query_is_definition_like and not query_is_structure_like:
                prior -= 0.2
        if query_is_definition_like and not query_is_structure_like and not looks_command_like(text):
            if looks_definition_text(text):
                prior += 0.22
            if looks_definition_text(section_text):
                prior += 0.08
        if query_is_structure_like:
            if looks_structure_text(text):
                prior += 0.12
            if looks_structure_text(section_text):
                prior += 0.12
        return prior
