from __future__ import annotations

from pkp.types.content import SourceType
from pkp.types.processing import ChunkingStrategy, ChunkRoutingDecision, DocumentFeatures


class ChunkRoutingService:
    def route(self, features: DocumentFeatures) -> ChunkRoutingDecision:
        reasons: list[str] = []
        debug = {
            "heading_quality_score": f"{features.heading_quality_score:.3f}",
            "section_count": str(features.section_count),
            "table_count": str(features.table_count),
            "figure_count": str(features.figure_count),
            "ocr_region_count": str(features.ocr_region_count),
        }
        if features.source_type is SourceType.PDF:
            reasons.append("PDF defaults to HybridChunker for mixed layout resilience.")
            if features.table_count > 0 or features.avg_section_words >= 220:
                reasons.append("Large or mixed-content PDF enables local refine.")
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=ChunkingStrategy.HYBRID,
                special_chunk_mode=True,
                local_refine=features.table_count > 0 or features.avg_section_words >= 220,
                fallback=False,
                reasons=reasons,
                debug=debug,
            )
        if features.source_type is SourceType.MARKDOWN:
            reasons.append("Markdown headings are usually reliable, prefer HierarchicalChunker.")
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=ChunkingStrategy.HIERARCHICAL,
                special_chunk_mode=True,
                local_refine=False,
                fallback=False,
                reasons=reasons,
                debug=debug,
            )
        if features.source_type is SourceType.DOCX:
            if features.heading_quality_score >= 0.55 and features.heading_count >= 2:
                reasons.append("DOCX heading quality is high, prefer HierarchicalChunker.")
                strategy = ChunkingStrategy.HIERARCHICAL
                local_refine = False
            else:
                reasons.append("DOCX heading quality is low, fall back to HybridChunker.")
                strategy = ChunkingStrategy.HYBRID
                local_refine = True
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=strategy,
                special_chunk_mode=True,
                local_refine=local_refine,
                fallback=False,
                reasons=reasons,
                debug=debug,
            )
        if features.source_type is SourceType.IMAGE:
            reasons.append("Image route uses regions, summary chunks, and caption binding.")
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=ChunkingStrategy.IMAGE,
                special_chunk_mode=True,
                local_refine=True,
                fallback=True,
                reasons=reasons,
                debug=debug,
            )
        raise ValueError(f"Unsupported source type for routing: {features.source_type}")
