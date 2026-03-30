from pathlib import Path

from rag import RAG, StorageConfig
from rag.query import QueryOptions


def main() -> None:
    core = RAG(storage=StorageConfig(root=Path(".rag-demo")))

    content_text="年假规定：入职满1年不满10年，每年5天带薪年假。入职满10年不满20年，每年10天。入职满20年以上，每年15天。年假需提前3个工作日申请。当年未使用的年假可顺延至次年3月31日，逾期作废。病假规定：需提供正规医院诊断证明。3天以内直属上级审批，3天以上部门总监审批。病假期间工资按基本工资的80%发放。连续病假超过30天按长期病假政策处理。事假规定：事假为无薪假期，按日扣除工资。每次不超过3天，需提前2个工作日申请。全年累计不超过15天。试用期规定：试用期为3个月，工资为正式工资的90%。第一个月末进行非正式1v1反馈。试用期满前两周进行转正评估。转正需准备PPT进行答辩。绩效考核：采用季度考核制。维度包括业务成果50%、专业能力20%、协作沟通15%、文化价值观15%。S级（前10%）年终奖系数2.0，A级（前30%）系数1.5，B级（中间50%）系数1.0，C级（后10%）无年终奖。"
    ingest_result = core.insert(
        source_type="plain_text",
        location="memory://note-1",
        owner="user",
        content_text=content_text,
    )

    query_result = core.query(
        "病假期间的工资如何发放？",
        options=QueryOptions(mode="mix"),
    )

    print("document_id:", ingest_result.document_id)
    print("chunk_count:", ingest_result.chunk_count)
    print("answer:", query_result.answer.answer_text)
    print("evidence:")
    for item in query_result.context.evidence[:3]:
        print(f"  - {item.citation_anchor} | score={item.score:.3f}")

    delete_result = core.delete(location="memory://note-1")
    print("deleted_doc_ids:", delete_result.deleted_doc_ids)


if __name__ == "__main__":
    main()
