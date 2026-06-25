import json
import time
import os
from pathlib import Path

import chromadb
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision, context_recall
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

load_dotenv()

# ── RAGAS Setup ───────────────────────────────────────────────
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o-mini", client=_openai_client)
embeddings = embedding_factory(client=_openai_client)

class EmbedQueryAdapter:
    def __init__(self, modern_embeddings):
        self._inner = modern_embeddings
    def embed_query(self, text): return self._inner.embed_text(text)
    def embed_documents(self, texts): return [self._inner.embed_text(t) for t in texts]
    def __getattr__(self, name): return getattr(self._inner, name)

answer_relevancy = AnswerRelevancy(embeddings=EmbedQueryAdapter(embeddings), llm=llm)
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# ── Load 10 Golden Questions ──────────────────────────────────
with open("golden_qa.json") as f:
    all_qs = json.load(f)["questions"]

# Select the first 10 factual/synthesis questions
qs = [q for q in all_qs if q["tier"] in [1, 2]][:10]

# Extract only the required papers to save compute
paper_ids = set()
for q in qs:
    paper_ids.update(q["relevant_paper_ids"])

PARSED_DIR = Path("data/parsed")
sidecars = []
for pid in paper_ids:
    path = PARSED_DIR / f"{pid}.json"
    if path.exists():
        with open(path) as f:
            sidecars.append(json.load(f))

# ── Setup Splitters ───────────────────────────────────────────
TOKENIZER = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int: return len(TOKENIZER.encode(text))

splitters = {
    "Fixed_256": CharacterTextSplitter(
        chunk_size=256, chunk_overlap=0, separator="", length_function=count_tokens
    ),
    "Fixed_512": CharacterTextSplitter(
        chunk_size=512, chunk_overlap=0, separator="", length_function=count_tokens
    ),
    "Recursive_512": RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""], length_function=count_tokens
    )
}

# ── Run Ablation ──────────────────────────────────────────────
print(f"Loaded {len(sidecars)} papers for 10 questions.")
print("Loading BGE model (this may take a moment)...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

results = {}

for name, splitter in splitters.items():
    print(f"\n{'='*50}\nTesting Strategy: {name}\n{'='*50}")
    
    # 1. Chunking
    chunks_text, chunk_metas = [], []
    for sidecar in sidecars:
        text = sidecar.get("full_text", "")
        if not text: continue
        raw_chunks = splitter.split_text(text)
        for idx, chunk in enumerate(raw_chunks):
            chunks_text.append(chunk)
            chunk_metas.append({"arxiv_id": sidecar["arxiv_id"], "chunk_index": idx, "title": sidecar["title"][:100]})
            
    # 2. Ephemeral DB (in-memory, fresh for each strategy)
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"temp_eval_{name}")
    
    print(f"Embedding {len(chunks_text)} chunks for {name}...")
    texts_with_prefix = [f"Represent this passage for retrieval: {t}" for t in chunks_text]
    embs = model.encode(texts_with_prefix, normalize_embeddings=True, show_progress_bar=False)
    
    ids = [f"{m['arxiv_id']}_{m['chunk_index']}" for m in chunk_metas]
    collection.upsert(ids=ids, embeddings=embs.tolist(), documents=chunks_text, metadatas=chunk_metas)
    
    # 3. Retrieval & Generation
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q in qs:
        q_emb = model.encode([f"Represent this sentence for searching: {q['question']}"], normalize_embeddings=True)
        res = collection.query(query_embeddings=q_emb.tolist(), n_results=5)
        
        contexts = res["documents"][0]
        context_block = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        prompt = f"Context:\n{context_block}\n\nQuestion: {q['question']}\n\nAnswer using only context."
        
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.1, 
            messages=[{"role": "user", "content": prompt}]
        )
        
        rows["question"].append(q["question"])
        rows["answer"].append(resp.choices[0].message.content)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(q["ground_truth"])
        
    # 4. Evaluation
    print(f"Running RAGAS evaluation for {name}...")
    ds = Dataset.from_dict(rows)
    eval_res = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
    scores = eval_res.to_pandas()[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()
    
    print(f"\nScores for {name}:")
    for k, v in scores.items(): 
        print(f"  {k:20s}: {v:.4f}")
    results[name] = scores

# ── Save Results ──────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "chunk_eval.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Chunking ablation complete! Saved to results/chunk_eval.json")