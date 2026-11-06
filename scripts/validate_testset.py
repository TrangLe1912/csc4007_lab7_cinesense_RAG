"""Kiểm tra bộ test multi-hop của sinh viên theo yêu cầu Bài 9 (>= 10 câu, >= 3 loại, >= 2 câu không trả lời trọn vẹn).

Dùng: python scripts/validate_testset.py data/multihop_testset.jsonl
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evaluation import read_jsonl  # noqa: E402
from src.retriever import load_chunks  # noqa: E402

TYPES = {"bridge", "comparison", "bridge-comparison", "intersection", "implicit", "single-hop"}
ANSWERABLE = {"full", "partial", "none"}
FIELDS = ("qid", "question", "type", "n_hops", "answerable", "gold_chain", "gold_answer", "gold_sp")


def validate(rows, kb_ids, min_questions=10, min_types=3, min_unanswerable=2):
    errors = []
    qids = [r.get("qid") for r in rows]
    if len(set(qids)) != len(qids):
        errors.append("qid bị trùng")
    for r in rows:
        q = r.get("qid", "?")
        missing = [k for k in FIELDS if k not in r]
        if missing:
            errors.append(f"{q}: thiếu trường {missing}")
            continue
        if r["type"] not in TYPES:
            errors.append(f"{q}: type '{r['type']}' không hợp lệ")
        if r["answerable"] not in ANSWERABLE:
            errors.append(f"{q}: answerable '{r['answerable']}' không hợp lệ")
        if r["n_hops"] != len(r["gold_chain"]):
            errors.append(f"{q}: n_hops != số hop trong gold_chain")
        chain_ids = [c for h in r["gold_chain"] for c in h["gold_chunk_ids"]]
        for c in chain_ids + list(r["gold_sp"]):
            if c not in kb_ids:
                errors.append(f"{q}: chunk_id không có trong KB: {c}")
        if set(chain_ids) != set(r["gold_sp"]):
            errors.append(f"{q}: gold_sp phải đúng bằng hợp các gold_chunk_ids trong gold_chain")
        has_gap = any(h["intermediate_answer"] == "NO_EVIDENCE" for h in r["gold_chain"])
        if r["answerable"] != "full" and not has_gap:
            errors.append(f"{q}: câu không trả lời trọn vẹn phải có hop NO_EVIDENCE")
        if r["answerable"] == "full" and has_gap:
            errors.append(f"{q}: answerable=full nhưng có hop NO_EVIDENCE")
    if len(rows) < min_questions:
        errors.append(f"Cần >= {min_questions} câu, hiện có {len(rows)}")
    if len({r.get("type") for r in rows}) < min_types:
        errors.append(f"Cần >= {min_types} loại câu hỏi")
    if sum(r.get("answerable") != "full" for r in rows) < min_unanswerable:
        errors.append(f"Cần >= {min_unanswerable} câu không trả lời được trọn vẹn")
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("--min-questions", type=int, default=10)
    ap.add_argument("--min-unanswerable", type=int, default=2)
    args = ap.parse_args()
    rows = read_jsonl(args.testset)
    errs = validate(rows, {c["chunk_id"] for c in load_chunks()}, min_questions=args.min_questions,
                    min_unanswerable=args.min_unanswerable)
    if errs:
        print("FAIL:")
        for e in errs:
            print(" -", e)
        sys.exit(1)
    print(f"PASS: {len(rows)} câu, {len({r['type'] for r in rows})} loại, "
          f"{sum(r['answerable'] != 'full' for r in rows)} câu không trả lời trọn vẹn")
