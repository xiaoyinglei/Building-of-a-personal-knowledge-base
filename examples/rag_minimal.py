from pathlib import Path

from rag import RAG, StorageConfig
from rag.query import QueryOptions


def main() -> None:
    core = RAG(storage=StorageConfig(root=Path(".rag-demo")))

    content_text="北新建材在“十五五”末，要实现营业收入1000亿元，利润总额100亿元，公司市值1000亿元的总体目标。在石膏板体系方面，加强高端零售产品占比，推挤聚焦装配式内装的体系能力与终端品牌实力建设，加快推进石膏板功能化、复合式、绿色化、装配式，推进装修产品化，形成完善的装配式石膏基饰面板产品及应用解决方案放。连续病假超过30天按长期病假政策处理。事假规定：事假为无薪假期，按日扣除工资。每次不超过3天，需提前2个工作日申请。全年累计不超过15天。试用期规定：试用期为3个月，工资为正式工资的90%。第一个月末进行非正式1v1反馈。试用期满前两周进行转正评估。转正需准备PPT进行答辩。绩效考核：采用季度考核制。维度包括业务成果50%、专业能力20%、协作沟通15%、文化价值观15%。S级（前10%）年终奖系数2.0，A级（前30%）系数1.5，B级（中间50%）系数1.0，C级（后10%）无年终奖。"
    ingest_result = core.insert(
        source_type="plain_text",
        location="memory://note-1",
        owner="user",
        content_text=content_text,
    )

    query_result = core.query(
        "北新建材要实现多少营收",
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
