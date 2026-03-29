from pkp.rerank.cross_encoder import CrossEncoderConfig, ProviderBackedCrossEncoder
from pkp.rerank.pipeline import FormalRerankService, RerankPipelineConfig
from pkp.service.rerank_service import HeuristicRerankService

CrossEncoderReranker = ProviderBackedCrossEncoder
FormalRerankPipeline = FormalRerankService

__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "FormalRerankPipeline",
    "FormalRerankService",
    "ProviderBackedCrossEncoder",
    "HeuristicRerankService",
    "RerankPipelineConfig",
]
