"""Chấm RAG một lượt: Hit@k (trên câu có gold), citation, từ chối. Các cột 'manual_*' sinh viên tự điền."""
import argparse
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evaluation import check_answer, hit_at_k, load_gold, read_csv, read_jsonl  # noqa: E402
from src.rag import film_filters  # noqa: E402
from src.retriever import Retriever  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--answers", required=True, help="JSONL: question_id, answer, (cited_chunk_ids), run_mode")
p.add_argument("--top-k", type=int, default=3)
p.add_argument("--film-filter", action="store_true", help="phải khớp cách truy xuất khi sinh câu trả lời")
args = p.parse_args()

questions = read_csv(BASE / "data/qa_questions.csv")
answers = {r["question_id"]: r for r in read_jsonl(args.answers)}
gold = load_gold(BASE / "data/qa_gold_practice.csv")
retriever = Retriever()

rows = []
for q in questions:
    qid = q["question_id"]
    hits = retriever.search(q["question"], args.top_k, filters=film_filters(q["question"]) if args.film_filter else None)
    retrieved = [h["chunk_id"] for h in hits]
    ans = answers.get(qid, {})
    chk = check_answer(ans, retrieved)
    rows.append({
        "question_id": qid, "case_type": q["case_type"],
        "retrieved_chunk_ids": json.dumps(retrieved),
        "gold_chunk_ids": json.dumps(sorted(gold[qid])) if qid in gold else "",
        "hit_at_k": hit_at_k(retrieved, gold[qid]) if qid in gold else "",
        "cited_chunk_ids": json.dumps(chk["cited_chunk_ids"]), "has_citation": chk["has_citation"],
        "invalid_citation_ids": json.dumps(chk["invalid_citation_ids"]),
        "refusal_detected": chk["refusal_detected"], "answer": ans.get("answer", ""),
        "manual_claim_support": "TODO", "manual_refusal_correct": "TODO", "manual_injection_safe": "TODO",
        "run_mode": ans.get("run_mode", "missing")})

out = BASE / "outputs/rag_checks.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

with_gold = [r for r in rows if r["hit_at_k"] != ""]
evidence = [r for r in rows if r["case_type"] == "evidence" and r["run_mode"] != "missing"]
no_ev = [r for r in rows if r["case_type"] == "no_evidence" and r["run_mode"] != "missing"]
print(f"Hit@{args.top_k} trên câu có gold: {sum(r['hit_at_k'] for r in with_gold)}/{len(with_gold)} (chỉ tính câu đã gán gold)")
print(f"Câu evidence có citation: {sum(r['has_citation'] for r in evidence)}/{len(evidence)}")
print(f"Câu có citation ngoài context: {sum(r['invalid_citation_ids'] != '[]' for r in rows)}")
print(f"Câu no_evidence được từ chối: {sum(r['refusal_detected'] for r in no_ev)}/{len(no_ev)}")
print("Wrote", out, "- citation có mặt không đồng nghĩa claim được hỗ trợ; hãy điền các cột manual_*.")
