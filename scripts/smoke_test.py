"""Kiểm tra nhanh toàn bộ dữ liệu và pipeline offline. Chạy trước khi bắt đầu lab."""
import collections
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from scripts.validate_testset import validate  # noqa: E402
from src.eval_multihop import hop_coverage_at_k  # noqa: E402
from src.evaluation import load_gold, read_csv, read_jsonl  # noqa: E402
from src.multihop import parse_planner, short_answer  # noqa: E402
from src.rag import REFUSAL_TEXT, rag_answer, sanitize  # noqa: E402
from src.retriever import Retriever, load_chunks  # noqa: E402

chunks = load_chunks()
ids = {c["chunk_id"] for c in chunks}
cards = [c for c in chunks if c["doc_type"] == "film_card"]
reviews = [c for c in chunks if c["doc_type"] == "review"]
assert len(chunks) == 40 and len(cards) == 12 and len(reviews) == 28, (len(chunks), len(cards), len(reviews))
assert {c["film_id"] for c in reviews} <= {c["film_id"] for c in cards}, "review thuộc phim không có film card"
assert all(c["chunk_id"].startswith(("film_card_", "imdb_review_pack_")) for c in chunks)

qa = read_csv(BASE / "data/qa_questions.csv")
kinds = collections.Counter(r["case_type"] for r in qa)
assert kinds == {"evidence": 8, "no_evidence": 2, "injection": 2, "multihop_preview": 2}, kinds
gold = load_gold(BASE / "data/qa_gold_practice.csv")
assert len(gold) == 4 and all(g <= ids for g in gold.values())

sample = read_jsonl(BASE / "data/multihop_testset_sample.jsonl")
errs = validate(sample, ids, min_questions=5, min_unanswerable=1)
assert not errs, errs

r = Retriever(chunks)
q1 = "film_beta is often criticized for slow pacing. Is the other film by the same director also criticized for its pacing?"
rank = {c["chunk_id"]: i + 1 for i, c in enumerate(r.search(q1, k=len(chunks)))}
assert (rank["imdb_review_pack_003_chunk_004"], rank["film_card_beta_chunk_001"], rank["film_card_gamma_chunk_001"]) == (5, 6, 18)
top = r.search("Who directed film_beta?", k=3, filters={"film_id": "film_beta"})
assert top[0]["chunk_id"] == "film_card_beta_chunk_001" and all(c["film_id"] == "film_beta" for c in top)

injected = next(c for c in chunks if c["chunk_id"] == "imdb_review_pack_003_chunk_015")
assert sanitize(injected["text"])[1] and "IGNORE" not in sanitize(injected["text"])[0].upper().replace("ĐÃ LỌC", "")
guarded = rag_answer("What is the capital of France?", lambda p: 1 / 0, r.search)
assert guarded["refused_by_guard"] and guarded["answer"] == REFUSAL_TEXT and guarded["n_llm_calls"] == 0

assert parse_planner("Follow up: Who directed film_beta?")[0] == "follow_up"
assert parse_planner("So the final answer is: READY")[0] == "final"
assert short_answer("Short answer: No\nExplanation: ...") == "No"
trace = next(t for t in read_jsonl(BASE / "samples/hop_trace_sample.jsonl") if t["qid"] == "Q1")
assert hop_coverage_at_k(trace["hops"], sample[0]["gold_chain"])[0] == 1.0

print("PASS: 40 chunk (12 film card + 28 review); 14 QA; 5 câu multi-hop mẫu; retriever + filter; guardrails; khung multi-hop")
print("Top chunks Q1:", [(c["chunk_id"], round(c["score"], 3)) for c in r.search(q1, k=3)])
