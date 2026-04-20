# L3/L4 Engine Design

**Date:** 2026-04-19

## Goal

在不破坏 `runtime.query()`、CLI、Workbench 现有同步外壳的前提下，重建一个更干净的 L3/L4 引擎内核：

- `planning_graph` 负责策略生成
- `retrieval_adapter` 负责多路召回编排
- `rerank_service` 负责候选清洗、硬截断、置信度审计
- `runtime_coordinator` 负责同步桥接和遗留胖对象兼容

## Core Principles

1. 结构上激进，零件上保守  
   继续复用 `QueryUnderstandingService`、`RoutingService`、`BranchRetrieverRegistry`、`EvidenceService`、`FormalRerankService` 等旧零件，但不再让它们直接决定 L3/L4 的主结构。

2. 内核异步，出口同步  
   新核心通过 `_async_retrieve_pipeline` 运行；同步 `retrieve()` 只在 `runtime_coordinator` 中做桥接。

3. 兼容层单独隔离  
   新引擎不直接拼装旧 `RetrievalResult`。兼容层通过独立的脂肪填充函数把 lean payload 变成历史胖对象。

## Architecture

### L3 `planning_graph`

输出 `PlanningState`，包含：

- `complexity_gate`
- `semantic_route`
- `target_collections`
- `predicate_plan`
- `retrieval_paths`
- `rewritten_query` / `sparse_query`

当前实现先采用同步规则节点的异步封装接口，后续如果引入真实 LangGraph 1.2，可直接把节点下沉为 `async def` 图节点。

### L4 `retrieval_adapter`

职责：

- 按 `PlanningState` 决定实际 branch 调用
- 统一 dense / sparse query 变体
- 控制大 scope 下的 whitelist / attribute filter 策略
- 保留 plan-aware retriever 扩展点，给 Milvus hybrid path 使用

### L4 `rerank_service`

职责：

- Pre-Rerank Protocol
- 强制 `max_model_candidates=50`
- 置信度审计与退出决策

## Compatibility Shim

`runtime_coordinator.inflate_legacy_retrieval_result()` 是唯一允许把 lean payload “变胖”的地方。

TODO:

- 当 CLI / Workbench 直接消费 lean async payload 后，删除该 shim。

## Remaining Work

1. 为 Milvus 2.4 增加真正的 hybrid retriever，并确保 `expr` 穿透到每个 `AnnSearchRequest`
2. 将 `planning_graph` 下沉到真实 LangGraph 1.2 状态图
3. 让 CLI / Workbench 直接走 async path，移除同步桥接

