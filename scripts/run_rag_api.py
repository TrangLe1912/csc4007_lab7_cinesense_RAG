"""Chế độ API (tuỳ chọn): chạy RAG một lượt qua endpoint OpenAI-compatible mà lớp được phép dùng.

Cần LLM_API_URL, LLM_API_KEY, LLM_MODEL trong biến môi trường (xem .env.example).
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evaluation import read_csv  # noqa: E402
from src.llm_adapter import call_llm  # noqa: E402
from src.rag import rag_answer  # noqa: E402
from src.retriever import retrieve  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--top-k", type=int, default=3)
p.add_argument("--film-filter", action="store_true")
args = p.parse_args()

out = BASE / "outputs/rag_answers_api.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for q in read_csv(BASE / "data/qa_questions.csv"):
        res = rag_answer(q["question"], call_llm, retrieve, k=args.top_k, use_film_filter=args.film_filter)
        entry = {"question_id": q["question_id"], **res, "top_k": args.top_k, "film_filter": args.film_filter,
                 "run_mode": "api_live_inference", "model": os.environ.get("LLM_MODEL", ""),
                 "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "notes": "cited_chunk_ids được tách tự động từ câu trả lời; vẫn phải kiểm tra thủ công claim có được hỗ trợ."}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("Wrote", out)
