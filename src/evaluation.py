"""Chấm RAG một lượt: Hit@k, kiểm tra citation, nhận diện từ chối. Giảng viên cung cấp sẵn."""
import csv
import json
from pathlib import Path

from src.rag import extract_citations, is_refusal


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_gold(path):
    """``question_id -> set(chunk_id)``. Chỉ các câu đã được gán gold mới có trong kết quả."""
    return {r["question_id"]: {c for c in r["gold_chunk_ids"].split(";") if c} for r in read_csv(path)}


def hit_at_k(retrieved_ids, gold_ids):
    """Top-k chứa ÍT NHẤT MỘT chunk vàng. Không bảo đảm đã thu thập đủ mọi bằng chứng."""
    return bool(set(retrieved_ids) & set(gold_ids))


def recall_at_k(retrieved_ids, gold_ids):
    gold = set(gold_ids)
    return len(gold & set(retrieved_ids)) / len(gold) if gold else 0.0


def check_answer(record, retrieved_ids):
    """Kiểm tra máy móc một bản ghi trả lời. Citation có mặt KHÔNG đồng nghĩa với việc claim được hỗ trợ."""
    answer = record.get("answer", "")
    cited = record.get("cited_chunk_ids") or extract_citations(answer)
    return {
        "cited_chunk_ids": cited,
        "has_citation": bool(cited),
        "invalid_citation_ids": [c for c in cited if c not in retrieved_ids],
        "refusal_detected": is_refusal(answer),
    }
