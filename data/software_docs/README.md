# Software docs corpus (secondary)

Original teaching notes used to prove the RAG pipeline is corpus-agnostic.
These are **not** a dump of FastAPI's website. RFCs are summarized from the
public-domain IETF documents. FastAPI / Flask / HTTP notes are written from
first principles so the second collection stays legally and technically
separate from the arXiv research corpus.

Ingest with:

```
python src/ingest.py --corpus software_docs
python src/chunk.py --corpus software_docs
python src/embed.py --corpus software_docs
python src/evaluate.py --corpus software_docs --golden_set golden_qa_software.json
```
