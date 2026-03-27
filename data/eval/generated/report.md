# 离线检索评估报告

## 总览
- 文档数：4
- 问题数：6
- Embedding Provider：`offline-lexical`
- Embedding Model：`lexical-256`
- Embedding Space：`offline-lexical::lexical-256`
- Vector Hit Rate：1.00
- FTS Hit Rate：0.83
- Runtime Hit Rate：1.00

## Chunk 质检
- Parent chunks：7
- Child chunks：8
- Special chunks：6
- 重复 searchable chunks：0
- 空白 chunks：0
- 过短 child chunks：0
- 过长 child chunks：0
- metadata 缺失 chunks：0

## 人工检查怎么做
1. 先看每个问题的 `runtime_top_k` 前 3 名，确认是否已经出现正确答案词。
2. 如果命中的是 child chunk，再看它的 `parent_text_preview`，确认 parent 回填后上下文是否更完整。
3. 表格题要确认命中的 special chunk 类型是 `table chunk`，而不是普通段落。
4. 图片题要分别确认：数值题命中 `ocr_region`，场景题命中 `image_summary`。
5. 抽样区里重点看长段落 child：如果 child 单独看不完整，但 parent 能补全，说明切分和回填接口是有效的。

## 抽样 Chunk
### quarterly-review.md / parent
- chunk_id：`0-2d86be55dada`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/quarterly-review.md#section-11f8e50461`
- text：收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。
- parent：N/A
- metadata：`{"location": "data/eval/generated/samples/quarterly-review.md", "source_type": "", "toc_path": "季度复盘 > 收入分析"}`

### quarterly-review.md / child
- chunk_id：`chunk-ad96a2c6e075`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/quarterly-review.md#section-11f8e50461`
- text：收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。
- parent：收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。
- metadata：`{"location": "data/eval/generated/samples/quarterly-review.md", "source_type": "", "toc_path": "季度复盘 > 收入分析"}`

### quarterly-review.md / special
- chunk_id：`1-a8a658180771`
- special_chunk_type：`table`
- citation_anchor：`data/eval/generated/samples/quarterly-review.md#section-11f8e50461`
- text：| 指标 | 数值 | |------|------| | 收入 | 1280 | | 订单 | 342 | | 续约率 | 91% |
- parent：N/A
- metadata：`{"bbox": "", "location": "data/eval/generated/samples/quarterly-review.md", "page_no": "", "source_type": "", "toc_path": "季度复盘 > 收入分析"}`

### operations-brief.docx / parent
- chunk_id：`0-bf3999c10335`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/operations-brief.docx#operations-brief-edb3f961f0`
- text：本周值班目标是减少回归成本。
- parent：N/A
- metadata：`{"location": "data/eval/generated/samples/operations-brief.docx", "source_type": "", "toc_path": "operations-brief > 运营周报"}`

### operations-brief.docx / child
- chunk_id：`chunk-d9f540e8e68d`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/operations-brief.docx#operations-brief-edb3f961f0`
- text：本周值班目标是减少回归成本。
- parent：本周值班目标是减少回归成本。
- metadata：`{"location": "data/eval/generated/samples/operations-brief.docx", "source_type": "", "toc_path": "operations-brief > 运营周报"}`

### operations-brief.docx / special
- chunk_id：`3-44a5867e9630`
- special_chunk_type：`table`
- citation_anchor：`data/eval/generated/samples/operations-brief.docx#operations-brief-edb3f961f0`
- text：| 指标 | 数值 | |-------|------| | 待处理告警 | 7 | | 自动修复 | 5 |
- parent：N/A
- metadata：`{"bbox": "", "location": "data/eval/generated/samples/operations-brief.docx", "page_no": "", "source_type": "", "toc_path": "operations-brief > 运营周报"}`

### research-notes.pdf / parent
- chunk_id：`0-1fe53ea3538e`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/research-notes.pdf#page-1-be7a7d1cc6`
- text：Research Notes Fast Path should answer direct questions with citations.
- parent：N/A
- metadata：`{"location": "data/eval/generated/samples/research-notes.pdf", "source_type": "", "toc_path": "research-notes"}`

### research-notes.pdf / child
- chunk_id：`chunk-6c8241d3ed1d`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/research-notes.pdf#page-1-be7a7d1cc6`
- text：Research Notes Fast Path should answer direct questions with citations.
- parent：Research Notes Fast Path should answer direct questions with citations.
- metadata：`{"location": "data/eval/generated/samples/research-notes.pdf", "source_type": "", "toc_path": "research-notes"}`

### dashboard-metrics.png / parent
- chunk_id：`0-a64a03988c1a`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/dashboard-metrics.png#dashboard-metrics-35563d7dcb`
- text：一个包含 KPI 卡片的运营仪表盘截图
- parent：N/A
- metadata：`{"location": "data/eval/generated/samples/dashboard-metrics.png", "source_type": "", "toc_path": "dashboard-metrics"}`

### dashboard-metrics.png / child
- chunk_id：`chunk-737f560e028e`
- special_chunk_type：`normal`
- citation_anchor：`data/eval/generated/samples/dashboard-metrics.png#dashboard-metrics-35563d7dcb`
- text：运营仪表盘 图2 收入 1280 订单 342
- parent：一个包含 KPI 卡片的运营仪表盘截图
- metadata：`{"location": "data/eval/generated/samples/dashboard-metrics.png", "source_type": "", "toc_path": "dashboard-metrics"}`

### dashboard-metrics.png / special
- chunk_id：`0-be8b9516327d`
- special_chunk_type：`ocr_region`
- citation_anchor：`data/eval/generated/samples/dashboard-metrics.png#dashboard-metrics-35563d7dcb`
- text：图2
- parent：N/A
- metadata：`{"bbox": "16.00,16.00,120.00,52.00", "location": "data/eval/generated/samples/dashboard-metrics.png", "page_no": "1", "region_index": "0", "source_type": "image", "toc_path": "dashboard-metrics"}`

### dashboard-metrics.png / special
- chunk_id：`1-d3614188eea0`
- special_chunk_type：`ocr_region`
- citation_anchor：`data/eval/generated/samples/dashboard-metrics.png#dashboard-metrics-35563d7dcb`
- text：收入 1280
- parent：N/A
- metadata：`{"bbox": "16.00,64.00,220.00,120.00", "location": "data/eval/generated/samples/dashboard-metrics.png", "page_no": "1", "region_index": "1", "source_type": "image", "toc_path": "dashboard-metrics"}`

## 问题结果
### markdown_revenue_reason
- question：收入增长的两个主要原因是什么？
- category：`child`
- expected_terms：`企业续约增长, 自助升级增加`
- corpus_has_expected_answer：`True`
- likely_issue：`ok`
- vector_hit / fts_hit / runtime_hit：`True` / `True` / `True`
- parent_backfill_improves：`False`

| kind | rank | role | special | score | matched_terms | parent_matched_terms | expected_hit | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| runtime | 1 | child |  | 1.065 | 企业续约增长, 自助升级增加 | 企业续约增长, 自助升级增加 | True | 收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。 |
| runtime | 2 | special | table | 0.752 |  |  | False | / 指标 / 数值 / /------/------/ / 收入 / 1280 / / 订单 / 342 / / 续约率 / 91% / |
| runtime | 3 | child |  | 0.475 |  |  | False | 团队在这个阶段会反复抽样检查 chunk 有没有被切坏，尤其是长段落和多句解释型内容，因为这些地方最容易把答案拆碎。 第二道门槛是 metadata 审计必须显示必填字段缺失为零，至少要覆盖 location、toc_path、chunk_role 和 content_hash。 只有两个门槛同时满足，部署窗口才会打开。 |
| runtime | 4 | child |  | 0.117 |  |  | False | 上线评审不是只看单个指标，而是看一整段闭环证据。 第一道门槛是离线评估包里的检索精度必须稳定在 95% 以上，而且连续两轮都不能回落。 评审会还要求把 child 命中、table 命中和 parent 回填的结果分别记录下来，避免只看一个漂亮的总体分数。 |
| vector | 1 | child |  | 0.321 | 企业续约增长, 自助升级增加 | 企业续约增长, 自助升级增加 | True | 收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。 |
| vector | 2 | child |  | 0.116 |  |  | False | 团队在这个阶段会反复抽样检查 chunk 有没有被切坏，尤其是长段落和多句解释型内容，因为这些地方最容易把答案拆碎。 第二道门槛是 metadata 审计必须显示必填字段缺失为零，至少要覆盖 location、toc_path、chunk_role 和 content_hash。 只有两个门槛同时满足，部署窗口才会打开。 |
| vector | 3 | special | table | 0.085 |  |  | False | / 指标 / 数值 / /------/------/ / 收入 / 1280 / / 订单 / 342 / / 续约率 / 91% / |
| vector | 4 | child |  | 0.076 |  |  | False | 上线评审不是只看单个指标，而是看一整段闭环证据。 第一道门槛是离线评估包里的检索精度必须稳定在 95% 以上，而且连续两轮都不能回落。 评审会还要求把 child 命中、table 命中和 parent 回填的结果分别记录下来，避免只看一个漂亮的总体分数。 |
| fts | 1 | child |  | 1.000 | 企业续约增长, 自助升级增加 | 企业续约增长, 自助升级增加 | True | 收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。 |
| fts | 2 | special | table | 1.000 |  |  | False | / 指标 / 数值 / /------/------/ / 收入 / 1280 / / 订单 / 342 / / 续约率 / 91% / |
| fts | 3 | child |  | 1.000 |  |  | False | 团队在这个阶段会反复抽样检查 chunk 有没有被切坏，尤其是长段落和多句解释型内容，因为这些地方最容易把答案拆碎。 第二道门槛是 metadata 审计必须显示必填字段缺失为零，至少要覆盖 location、toc_path、chunk_role 和 content_hash。 只有两个门槛同时满足，部署窗口才会打开。 |

### markdown_release_gates
- question：发布前必须同时满足哪两个门槛？
- category：`parent_backfill`
- expected_terms：`95%, 必填字段缺失为零`
- corpus_has_expected_answer：`True`
- likely_issue：`parent_context_needed`
- vector_hit / fts_hit / runtime_hit：`True` / `True` / `True`
- parent_backfill_improves：`True`

| kind | rank | role | special | score | matched_terms | parent_matched_terms | expected_hit | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| runtime | 1 | child |  | 1.049 | 必填字段缺失为零 | 95%, 必填字段缺失为零 | True | 团队在这个阶段会反复抽样检查 chunk 有没有被切坏，尤其是长段落和多句解释型内容，因为这些地方最容易把答案拆碎。 第二道门槛是 metadata 审计必须显示必填字段缺失为零，至少要覆盖 location、toc_path、chunk_role 和 content_hash。 只有两个门槛同时满足，部署窗口才会打开。 |
| runtime | 2 | child |  | 0.991 | 95% | 95%, 必填字段缺失为零 | True | 上线评审不是只看单个指标，而是看一整段闭环证据。 第一道门槛是离线评估包里的检索精度必须稳定在 95% 以上，而且连续两轮都不能回落。 评审会还要求把 child 命中、table 命中和 parent 回填的结果分别记录下来，避免只看一个漂亮的总体分数。 |
| runtime | 3 | child |  | 0.549 |  |  | False | 收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。 |
| runtime | 4 | special | table | 0.138 |  |  | False | / 指标 / 数值 / /------/------/ / 收入 / 1280 / / 订单 / 342 / / 续约率 / 91% / |
| vector | 1 | child |  | 0.232 | 必填字段缺失为零 | 95%, 必填字段缺失为零 | True | 团队在这个阶段会反复抽样检查 chunk 有没有被切坏，尤其是长段落和多句解释型内容，因为这些地方最容易把答案拆碎。 第二道门槛是 metadata 审计必须显示必填字段缺失为零，至少要覆盖 location、toc_path、chunk_role 和 content_hash。 只有两个门槛同时满足，部署窗口才会打开。 |
| vector | 2 | child |  | 0.202 | 95% | 95%, 必填字段缺失为零 | True | 上线评审不是只看单个指标，而是看一整段闭环证据。 第一道门槛是离线评估包里的检索精度必须稳定在 95% 以上，而且连续两轮都不能回落。 评审会还要求把 child 命中、table 命中和 parent 回填的结果分别记录下来，避免只看一个漂亮的总体分数。 |
| vector | 3 | child |  | 0.138 |  |  | False | 收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。 |
| vector | 4 | special | table | 0.085 |  |  | False | / 指标 / 数值 / /------/------/ / 收入 / 1280 / / 订单 / 342 / / 续约率 / 91% / |
| fts | 1 | child |  | 1.000 | 必填字段缺失为零 | 95%, 必填字段缺失为零 | True | 团队在这个阶段会反复抽样检查 chunk 有没有被切坏，尤其是长段落和多句解释型内容，因为这些地方最容易把答案拆碎。 第二道门槛是 metadata 审计必须显示必填字段缺失为零，至少要覆盖 location、toc_path、chunk_role 和 content_hash。 只有两个门槛同时满足，部署窗口才会打开。 |
| fts | 2 | child |  | 1.000 | 95% | 95%, 必填字段缺失为零 | True | 上线评审不是只看单个指标，而是看一整段闭环证据。 第一道门槛是离线评估包里的检索精度必须稳定在 95% 以上，而且连续两轮都不能回落。 评审会还要求把 child 命中、table 命中和 parent 回填的结果分别记录下来，避免只看一个漂亮的总体分数。 |
| fts | 3 | child |  | 1.000 |  |  | False | 收入增长主要来自两个原因。第一，企业续约增长了，客服把续约流程从五步压缩到两步后，签回周期更短。第二，自助升级增加了，计费页面改版后，试用用户更容易完成升级。财务团队补充说，这两项变化同时作用在老客户和新试点项目上。 |

### docx_alert_table
- question：值班指标表里待处理告警是多少？
- category：`table`
- expected_terms：`7`
- corpus_has_expected_answer：`True`
- likely_issue：`ok`
- vector_hit / fts_hit / runtime_hit：`True` / `True` / `True`
- parent_backfill_improves：`False`

| kind | rank | role | special | score | matched_terms | parent_matched_terms | expected_hit | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| runtime | 1 | special | table | 0.714 | 7 |  | True | / 指标 / 数值 / /-------/------/ / 待处理告警 / 7 / / 自动修复 / 5 / |
| runtime | 2 | child |  | 0.773 |  |  | False | 运营组把凌晨告警清理时间从 42 分钟压缩到了 18 分钟，主要靠统一告警标签和更短的回滚手册。 |
| runtime | 3 | child |  | 0.618 |  |  | False | 本周值班目标是减少回归成本。 |
| vector | 1 | special | table | 0.371 | 7 |  | True | / 指标 / 数值 / /-------/------/ / 待处理告警 / 7 / / 自动修复 / 5 / |
| vector | 2 | child |  | 0.296 |  |  | False | 本周值班目标是减少回归成本。 |
| vector | 3 | child |  | 0.078 |  |  | False | 运营组把凌晨告警清理时间从 42 分钟压缩到了 18 分钟，主要靠统一告警标签和更短的回滚手册。 |
| fts | 1 | special | table | 1.000 | 7 |  | True | / 指标 / 数值 / /-------/------/ / 待处理告警 / 7 / / 自动修复 / 5 / |
| fts | 2 | child |  | 1.000 |  |  | False | 运营组把凌晨告警清理时间从 42 分钟压缩到了 18 分钟，主要靠统一告警标签和更短的回滚手册。 |
| fts | 3 | child |  | 1.000 |  |  | False | 本周值班目标是减少回归成本。 |

### pdf_deep_path_conflict
- question：What should Deep Path expose during research?
- category：`child`
- expected_terms：`expose conflicts`
- corpus_has_expected_answer：`True`
- likely_issue：`ok`
- vector_hit / fts_hit / runtime_hit：`True` / `True` / `True`
- parent_backfill_improves：`False`

| kind | rank | role | special | score | matched_terms | parent_matched_terms | expected_hit | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| runtime | 1 | child |  | 0.985 | expose conflicts | expose conflicts | True | Deep Path should decompose research questions, expose conflicts, and report uncertainty. Risk review says supply chain delay remains manageable when fallback suppliers are active. |
| runtime | 2 | child |  | 1.034 |  |  | False | Research Notes Fast Path should answer direct questions with citations. |
| runtime | 3 | child |  | 0.800 | expose conflicts | expose conflicts | True | Deep Path should decompose research questions, expose conflicts, and report uncertainty. Risk review says supply chain delay remains manageable when fallback suppliers are active. |
| runtime | 4 | child |  | 0.700 |  |  | False | Research Notes Fast Path should answer direct questions with citations. |
| vector | 1 | child |  | 0.478 |  |  | False | Research Notes Fast Path should answer direct questions with citations. |
| vector | 2 | child |  | 0.386 | expose conflicts | expose conflicts | True | Deep Path should decompose research questions, expose conflicts, and report uncertainty. Risk review says supply chain delay remains manageable when fallback suppliers are active. |
| fts | 1 | child |  | 1.000 | expose conflicts | expose conflicts | True | Deep Path should decompose research questions, expose conflicts, and report uncertainty. Risk review says supply chain delay remains manageable when fallback suppliers are active. |
| fts | 2 | child |  | 1.000 |  |  | False | Research Notes Fast Path should answer direct questions with citations. |

### image_ocr_revenue
- question：图里的收入数值是多少？
- category：`ocr`
- expected_terms：`1280`
- corpus_has_expected_answer：`True`
- likely_issue：`ok`
- vector_hit / fts_hit / runtime_hit：`True` / `True` / `True`
- parent_backfill_improves：`False`

| kind | rank | role | special | score | matched_terms | parent_matched_terms | expected_hit | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| runtime | 1 | special | ocr_region | 0.714 | 1280 |  | True | 收入 1280 |
| runtime | 2 | child |  | 0.650 | 1280 |  | False | 运营仪表盘 图2 收入 1280 订单 342 |
| runtime | 3 | special | ocr_region | 0.017 |  |  | False | 图2 |
| runtime | 4 | special | ocr_region | 0.017 |  |  | False | 订单 342 |
| runtime | 5 | special | image_summary | 0.017 |  |  | False | 一个包含 KPI 卡片的运营仪表盘截图 |
| vector | 1 | special | ocr_region | 0.224 | 1280 |  | True | 收入 1280 |
| vector | 2 | child |  | 0.191 | 1280 |  | False | 运营仪表盘 图2 收入 1280 订单 342 |
| vector | 3 | special | ocr_region | 0.000 |  |  | False | 图2 |
| vector | 4 | special | ocr_region | 0.000 |  |  | False | 订单 342 |
| vector | 5 | special | image_summary | 0.000 |  |  | False | 一个包含 KPI 卡片的运营仪表盘截图 |
| fts | 1 | special | ocr_region | 1.000 | 1280 |  | True | 收入 1280 |
| fts | 2 | child |  | 1.000 | 1280 |  | False | 运营仪表盘 图2 收入 1280 订单 342 |

### image_summary_scene
- question：这张图是什么类型的画面？
- category：`image_summary`
- expected_terms：`运营仪表盘截图`
- corpus_has_expected_answer：`True`
- likely_issue：`parent_context_needed`
- vector_hit / fts_hit / runtime_hit：`True` / `False` / `True`
- parent_backfill_improves：`True`

| kind | rank | role | special | score | matched_terms | parent_matched_terms | expected_hit | preview |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| runtime | 1 | special | image_summary | 0.427 | 运营仪表盘截图 |  | True | 一个包含 KPI 卡片的运营仪表盘截图 |
| runtime | 2 | special | ocr_region | 0.022 |  |  | False | 图2 |
| runtime | 3 | special | ocr_region | 0.021 |  |  | False | 收入 1280 |
| runtime | 4 | special | ocr_region | 0.021 |  |  | False | 订单 342 |
| runtime | 5 | child |  | 0.021 |  | 运营仪表盘截图 | False | 运营仪表盘 图2 收入 1280 订单 342 |
| vector | 1 | special | image_summary | 0.078 | 运营仪表盘截图 |  | True | 一个包含 KPI 卡片的运营仪表盘截图 |
| vector | 2 | special | ocr_region | 0.000 |  |  | False | 图2 |
| vector | 3 | special | ocr_region | 0.000 |  |  | False | 收入 1280 |
| vector | 4 | special | ocr_region | 0.000 |  |  | False | 订单 342 |
| vector | 5 | child |  | 0.000 |  | 运营仪表盘截图 | False | 运营仪表盘 图2 收入 1280 订单 342 |
