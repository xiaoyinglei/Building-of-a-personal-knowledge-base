# Corpus Test Notes

## arxiv_attention_is_all_you_need
- Best for: chunking long academic prose, retrieving by section name, reranking architecture details, figure and table lookup, citation checks.
- Expected question styles: abstract summary, encoder/decoder structure, complexity comparison from Table 1, BLEU lookup from Table 2.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: no
- table handling: yes
- citation: yes

## arxiv_retrieval_augmented_generation
- Best for: RAG-specific retrieval tests, abstract-level summarization, figure-based pipeline questions, score lookup from benchmark tables.
- Expected question styles: what memory is used, what retrieval method is used, which RAG variant wins on a metric, how the model is described.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: no
- table handling: yes
- citation: yes

## github_huggingface_transformers_readme
- Best for: markdown chunking, heading-aware retrieval, install-command lookup, quickstart code retrieval, short-answer generation.
- Expected question styles: minimum version requirements, where Pipeline is introduced, example model names, README summary.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: no
- table handling: limited
- citation: yes

## github_langchain_readme
- Best for: README-style retrieval, heading and bullet-list chunking, ecosystem entity disambiguation, short summaries.
- Expected question styles: install commands, ecosystem products, documentation entry points, why-use-LangChain summary.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: no
- table handling: no
- citation: yes

## nist_ai_rmf_1_0
- Best for: long formal PDF chunking, section and appendix retrieval, policy-language rerank, table lookup by function name.
- Expected question styles: core functions, table number lookup, appendix lookup, update/versioning description.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: no
- table handling: yes
- citation: yes

## nist_generative_ai_profile_600_1
- Best for: long policy-report chunking, section boundary detection, summary generation, appendix lookup, exact page hints from TOC.
- Expected question styles: executive-order basis, page lookup from TOC, four primary considerations, overview of GAI risks.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: no
- table handling: limited
- citation: yes

## hf_transformers_question_answering_doc
- Best for: technical-tutorial retrieval, code-block chunking, parameter lookup, procedural question answering.
- Expected question styles: model/dataset pairing, preprocessing arguments, training steps, inference workflow.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: no
- table handling: no
- citation: yes

## hf_transformers_document_question_answering_doc
- Best for: multimodal doc retrieval, OCR-pipeline retrieval, dependency lookup, long code-block chunking.
- Expected question styles: checkpoint name, OCR engine, dataset fields, modality summary, max-position limit discussion.
- chunking: yes
- retrieval: yes
- rerank: yes
- OCR: yes
- table handling: no
- citation: yes

## docvqa_letter_date_train_0
- Best for: OCR extraction, noisy scan handling, short-span retrieval, metadata alignment between image and answer key.
- Expected question styles: date extraction, contact name extraction, routing/instruction lookup from header text.
- chunking: limited
- retrieval: yes
- rerank: yes
- OCR: yes
- table handling: no
- citation: no

## docvqa_interoffice_correspondence_train_11
- Best for: OCR extraction on denser document text, header-field retrieval, subject-line lookup, document-type classification from scan text.
- Expected question styles: document type, location extraction, date extraction, subject extraction.
- chunking: limited
- retrieval: yes
- rerank: yes
- OCR: yes
- table handling: no
- citation: no
