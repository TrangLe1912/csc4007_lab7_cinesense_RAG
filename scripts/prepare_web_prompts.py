"""Chế độ Web UI: tạo mỗi câu hỏi một file prompt hoàn chỉnh để dán vào giao diện chatbot.

Không tự động hoá website bằng script này. Lưu phản hồi nguyên văn vào JSONL đúng schema trong samples/.
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evaluation import read_csv  # noqa: E402
from src.rag import build_prompt, film_filters  # noqa: E402
from src.retriever import Retriever  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--top-k", type=int, default=3)
p.add_argument("--film-filter", action="store_true")
args = p.parse_args()

out = BASE / "outputs/web_prompts"
out.mkdir(parents=True, exist_ok=True)
retriever = Retriever()
for q in read_csv(BASE / "data/qa_questions.csv"):
    hits = retriever.search(q["question"], args.top_k,
                            filters=film_filters(q["question"]) if args.film_filter else None)
    (out / f"RAG_{q['question_id']}.txt").write_text(build_prompt(q["question"], hits), encoding="utf-8")
print("Wrote prompts to", out, "- ghi phản hồi nguyên văn vào JSONL theo schema trong samples/rag_answers_sample.jsonl")
