# Minimal RAG Test Corpus

This corpus is a small but broad evaluation set for a personal knowledge-base RAG project. It contains 10 primary source documents across PDF papers, Markdown, formal reports, technical docs, and OCR document images.

## What was downloaded

- `arxiv_attention_is_all_you_need`: classic academic PDF with figures, tables, and long sections.
- `arxiv_retrieval_augmented_generation`: RAG-focused academic PDF with figure and benchmark tables.
- `github_huggingface_transformers_readme`: high-signal README with headings, lists, and code blocks.
- `github_langchain_readme`: framework README with quickstart, product overview, and documentation entry points.
- `nist_ai_rmf_1_0`: formal NIST PDF with TOC, tables, appendices, and policy language.
- `nist_generative_ai_profile_600_1`: formal NIST PDF with TOC, long sections, appendix structure, and GAI-specific risk guidance.
- `hf_transformers_question_answering_doc`: official technical tutorial for text question answering.
- `hf_transformers_document_question_answering_doc`: official technical tutorial for document visual question answering and OCR pipelines.
- `docvqa_letter_date_train_0`: scanned-like document image with accompanying metadata JSON.
- `docvqa_interoffice_correspondence_train_11`: scanned-like document image with accompanying metadata JSON.

## Why these documents

- The two arXiv papers give you structured academic PDFs with abstracts, numbered sections, figures, tables, and citation-friendly answers.
- The two GitHub Markdown files give you clean heading structure plus lists and code blocks, which is useful for testing markdown parsing and chunk boundaries.
- The two NIST reports give you long, formal PDFs with tables of contents, appendices, and policy prose that stress retrieval and reranking over longer spans.
- The two Hugging Face docs give you technical tutorial content with concrete commands, parameters, code blocks, and task definitions.
- The two DocVQA images give you OCR-specific evaluation cases with short spans, noisy scans, and paired metadata.

## Capability coverage

- Chunking: academic papers, NIST reports, and step-by-step technical docs all contain strong section boundaries and long passages.
- Retrieval: README headings, report TOCs, tutorial sections, and OCR header fields support direct section and field lookup.
- Rerank: papers and reports contain overlapping AI terminology, which is useful for distinguishing near-miss results.
- Context assembly: report sections and tutorials need multi-paragraph stitching to answer summary questions well.
- Answer generation: academic and technical docs support factual, summary, and citation-oriented responses.
- OCR: the two DocVQA samples specifically exercise image ingestion, OCR extraction, and header or field resolution.
- Table and figure handling: the academic papers and AI RMF report include retrievable figures and tables for structured questions.

## Files

- `manifest.json`: source inventory for the 10 primary corpus items.
- `doc_notes/corpus_test_notes.md`: per-document testing guidance and suitability notes.
- `eval_questions.jsonl`: 30 questions for manual debugging or automated evaluation.
- `ocr_images/*.json`: source metadata and original question-answer pairs for the OCR samples.

## How to use the evaluation questions

1. Ingest the files under `pdf/`, `markdown/`, `reports/`, `tech_docs/`, and `ocr_images/`.
2. Preserve the `doc_id` mapping from `manifest.json`.
3. Run retrieval and generation with each row in `eval_questions.jsonl`.
4. Compare retrieved documents against `expected_doc_id`.
5. Use `expected_section_hint` to inspect chunking and ranking errors.
6. Use `expected_answer_hint` as a lightweight answer rubric for manual review or weak automatic scoring.

## Suggested import flow for your knowledge base

1. Import all PDFs as binary source files and keep original filenames.
2. Import Markdown docs as text-first sources and preserve headings in chunk metadata.
3. Import OCR images together with their JSON metadata so you can test image-only, OCR-text-only, and hybrid retrieval paths.
4. Add `doc_id`, `source_type`, `format`, and `source_url` from `manifest.json` into your document metadata store.
5. Run one pass with chunk-level citations enabled so you can inspect whether questions map back to the expected section hints.

## Recommended first checks

- Start with the README and Hugging Face docs to validate markdown parsing and code-block chunking.
- Move to the two arXiv papers to test figure or table retrieval and citation grounding.
- Use the NIST reports to test longer-context reranking.
- Finish with the two OCR images to validate image ingestion and OCR extraction quality.
