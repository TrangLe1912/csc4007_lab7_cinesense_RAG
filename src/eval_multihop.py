"""Đánh giá 4 tầng cho multi-hop (Bài 9): đáp án, bằng chứng, chuỗi hop, chi phí.

Giảng viên cung cấp sẵn. Sinh viên đọc code để hiểu độ đo, không cần sửa.
"""
import collections
import re
import string

PUNCT = set(string.punctuation)


def normalize(s):
    """Chuẩn hoá kiểu SQuAD/HotpotQA: chữ thường, bỏ dấu câu, bỏ mạo từ a/an/the."""
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def answer_prf(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0, 0.0, 0.0
    P, R = same / len(p), same / len(g)
    return P, R, 2 * P * R / (P + R)


def sp_prf(pred_ids, gold_ids):
    """Supporting facts ở mức chunk (HotpotQA gốc dùng mức câu)."""
    pred, gold = set(pred_ids), set(gold_ids)
    tp = len(pred & gold)
    P = tp / len(pred) if pred else 0.0
    R = tp / len(gold) if gold else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def joint_prf(ans_prf, sp):
    """Joint theo HotpotQA: P_joint = P_ans * P_sp ; R_joint = R_ans * R_sp."""
    P, R = ans_prf[0] * sp[0], ans_prf[1] * sp[1]
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def hop_coverage_at_k(trace_hops, gold_chain, k=5):
    """Tỷ lệ hop vàng có chunk vàng nằm trong top-k của ÍT NHẤT MỘT lần truy xuất."""
    seen = set()
    for h in trace_hops:
        seen.update(r["chunk_id"] for r in h["retrieved"][:k])
    hops = [g for g in gold_chain if g["gold_chunk_ids"]]  # bỏ qua hop vàng "không có bằng chứng"
    hits = [int(any(cid in seen for cid in g["gold_chunk_ids"])) for g in hops]
    coverage = sum(hits) / len(hits) if hits else 0.0
    chain_recall = float(all(hits)) if hits else 0.0
    return coverage, chain_recall, hits
