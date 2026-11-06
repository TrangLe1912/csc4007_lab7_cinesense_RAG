"""Chạy retriever trên bộ câu hỏi QA và lưu top-k ra CSV (offline, không cần LLM)."""
import argparse
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evaluation import read_csv  # noqa: E402
from src.rag import film_filters  # noqa: E402
from src.retriever import Retriever  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--top-k", type=int, default=3)
p.add_argument("--film-filter", action="store_true",
               help="lọc metadata theo film_id nếu câu hỏi nêu tên phim (ví dụ film_beta)")
args = p.parse_args()

retriever = Retriever()
rows = []
for q in read_csv(BASE / "data/qa_questions.csv"):
    hits = retriever.search(q["question"], args.top_k, filters=film_filters(q["question"]) if args.film_filter else None)
    rows.append({"question_id": q["question_id"], "case_type": q["case_type"], "top_k": args.top_k,
                 "film_filter": args.film_filter,
                 "retrieved_chunk_ids": json.dumps([h["chunk_id"] for h in hits]),
                 "scores": json.dumps([round(h["score"], 3) for h in hits])})
suffix = "_filter" if args.film_filter else ""
out = BASE / f"outputs/retrieval_top{args.top_k}{suffix}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("Wrote", out)
