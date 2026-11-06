"""Ghép context (đã lọc injection) cho từng câu hỏi để dùng với API hoặc Web UI. Không gọi LLM."""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evaluation import read_csv  # noqa: E402
from src.rag import film_filters, format_context, sanitize  # noqa: E402
from src.retriever import Retriever  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--top-k", type=int, default=3)
p.add_argument("--film-filter", action="store_true")
args = p.parse_args()

retriever = Retriever()
out = BASE / "outputs/rag_contexts.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for q in read_csv(BASE / "data/qa_questions.csv"):
        hits = retriever.search(q["question"], args.top_k,
                                filters=film_filters(q["question"]) if args.film_filter else None)
        entry = {"question_id": q["question_id"], "question": q["question"], "case_type": q["case_type"],
                 "top_k": args.top_k, "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
                 "injection_flagged_chunk_ids": [h["chunk_id"] for h in hits if sanitize(h["text"])[1]],
                 "context": format_context(hits), "run_mode": "retrieval_only"}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("Wrote", out)
