"""Đánh giá 4 tầng và so sánh các hệ thống từ các file nhật ký (offline, không cần LLM).

Ví dụ:
    python scripts/evaluate_multihop.py --testset data/multihop_testset_sample.jsonl \\
        --runs "Multi-hop (mẫu)=samples/hop_trace_sample.jsonl"
    python scripts/evaluate_multihop.py --testset data/multihop_testset.jsonl \\
        --runs "Single-hop RAG=outputs/singlehop_runs.jsonl" "Multi-hop=outputs/hop_trace_log.jsonl"

Chỉ chấm các qid có mặt ở cả bộ test lẫn file chạy; cột N cho biết số câu được chấm.
"""
import argparse
import sys
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.eval_multihop import answer_em, answer_prf, hop_coverage_at_k, joint_prf, sp_prf  # noqa: E402
from src.evaluation import read_jsonl  # noqa: E402
from src.multihop import short_answer  # noqa: E402
from src.rag import CHUNK_ID_RE  # noqa: E402

TOP_K = 5


def score(run, gold, k=TOP_K):
    pred_short = short_answer(run["final_answer"])
    pred_sp = set(CHUNK_ID_RE.findall(run["final_answer"]))
    for h in run["hops"]:
        pred_sp.update(h.get("citations", []))
    a = answer_prf(pred_short, gold["gold_answer"])
    s = sp_prf(pred_sp, gold["gold_sp"])
    j = joint_prf(a, s)
    cov, chain, _ = hop_coverage_at_k(run["hops"], gold["gold_chain"], k)
    return {"em": answer_em(pred_short, gold["gold_answer"]), "f1": a[2], "sp_f1": s[2], "joint_f1": j[2],
            "coverage": cov, "chain": chain, "retr": run["n_retrieval_calls"], "llm": run["n_llm_calls"],
            "latency": run.get("latency_s"), "partial": float(run["status"] != "answered")}


COLUMNS = [("em", "Answer EM"), ("f1", "Answer F1"), ("sp_f1", "SP F1"), ("joint_f1", "Joint F1"),
           ("coverage", "Hop coverage@5"), ("chain", "Chain recall@5"), ("retr", "#Truy xuất/câu"),
           ("llm", "#Gọi LLM/câu"), ("latency", "Latency TB (s)"), ("partial", "Tỷ lệ trả lời một phần")]


def fmt(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{mean(vals):.2f}" if vals else "n/a"


def evaluate(testset_path, runs, out_md):
    gold = {g["qid"]: g for g in read_jsonl(testset_path)}
    table = {}
    for name, path in runs:
        recs = {r["qid"]: r for r in read_jsonl(path)}
        scored = [score(recs[q], g) for q, g in gold.items() if q in recs]
        if scored:
            table[name] = scored
    header = "| Hệ thống | N | " + " | ".join(c[1] for c in COLUMNS) + " |"
    lines = [header, "|---" * (len(COLUMNS) + 2) + "|"]
    for name, rs in table.items():
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(fmt(rs, c) for c, _ in COLUMNS) + " |")
    text = "# Single-hop vs Multi-hop\n\n" + "\n".join(lines) + "\n"
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help='"Tên=đường/dẫn.jsonl" (có thể nhiều hệ thống)')
    ap.add_argument("--out", default=str(BASE / "outputs/singlehop_vs_multihop.md"))
    args = ap.parse_args()
    pairs = []
    for item in args.runs:
        name, _, path = item.partition("=")
        pairs.append((name, path) if path else (Path(name).stem, name))
    print(evaluate(args.testset, pairs, args.out))
    print("Wrote", args.out)
