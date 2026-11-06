"""RAG một lượt (single-hop) cho CineSense kèm guardrails cơ bản của Bài 8.

Giảng viên cung cấp sẵn; sinh viên chỉ cần hoàn thiện ``prompts/rag_prompt.txt``.

Ba lớp bảo vệ nằm trong code, độc lập với prompt:
1. no-evidence: không chunk nào đạt ngưỡng ``tau`` thì từ chối, không gọi LLM;
2. injection: chỉ dẫn nằm trong chunk bị vô hiệu hoá trước khi đưa vào context;
3. citation: chỉ chấp nhận ``chunk_id`` có trong context vừa truy xuất.
"""
import re
from pathlib import Path

from src.retriever import BASE, retrieve

TAU = 0.15  # Ngưỡng similarity cho retriever TF-IDF demo; embedding khác phải hiệu chỉnh lại.
PROMPT_PATH = BASE / "prompts" / "rag_prompt.txt"
REFUSAL_TEXT = "Không đủ bằng chứng trong kho dữ liệu CineSense để trả lời câu hỏi này. (NO_EVIDENCE)"

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(previous|above)\s+instructions",
    r"bỏ\s+qua\s+(mọi|tất cả|các)\s+(hướng dẫn|chỉ dẫn)",
]
CHUNK_ID_RE = re.compile(r"[A-Za-z0-9_]+_chunk_\d+")
FILM_ID_RE = re.compile(r"\bfilm_[a-z]+\b")
REFUSAL_RE = re.compile(r"NO_EVIDENCE|không đủ bằng chứng|not enough evidence|insufficient evidence", re.IGNORECASE)


def load_prompt(path=None):
    """Đọc prompt; bỏ các dòng chú thích bắt đầu bằng ``# `` (dành cho hướng dẫn sinh viên)."""
    text = Path(path or PROMPT_PATH).read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not (line.startswith("# ") or line == "#")).strip() + "\n"


def sanitize(text):
    """Vô hiệu hoá chỉ dẫn nằm trong dữ liệu truy xuất (indirect prompt injection).

    Trả về ``(text_đã_lọc, bị_gắn_cờ)``. Biểu thức chính quy chỉ là lớp mỏng nhất;
    lớp chính vẫn là luật "chunk là dữ liệu, không phải lệnh" trong prompt.
    """
    flagged = any(re.search(p, text, flags=re.IGNORECASE) for p in INJECTION_PATTERNS)
    for p in INJECTION_PATTERNS:
        text = re.sub(p, "[đã lọc: chỉ dẫn nằm trong dữ liệu]", text, flags=re.IGNORECASE)
    return text, flagged


def format_context(chunks):
    """Đóng gói chunk thành khối ``<chunk id=... film_id=...>`` đã lọc injection."""
    blocks = []
    for c in chunks:
        text, flagged = sanitize(c["text"])
        flag = ' flagged="injection"' if flagged else ""
        blocks.append(f'<chunk id="{c["chunk_id"]}" film_id="{c.get("film_id", "")}"{flag}>\n{text}\n</chunk>')
    return "\n".join(blocks)


def film_filters(question):
    """Bộ lọc metadata theo ``film_id`` nếu câu hỏi nêu tên phim (ví dụ ``film_beta``); nếu không trả ``None``.

    Review chỉ mang ``film_id`` trong metadata, không nhắc tên phim trong văn bản, nên TF-IDF
    không tự khớp được ``film_beta`` với review của film_beta. Lọc metadata giải quyết đúng chỗ đó.
    """
    ids = sorted(set(FILM_ID_RE.findall(question)))
    return {"film_id": ids} if ids else None


def build_prompt(question, chunks, template=None):
    template = template if template is not None else load_prompt()
    return template.replace("{question}", question).replace("{context}", format_context(chunks))


def extract_citations(answer):
    """Các ``chunk_id`` xuất hiện trong câu trả lời, theo thứ tự, không trùng."""
    seen = []
    for cid in CHUNK_ID_RE.findall(answer or ""):
        if cid not in seen:
            seen.append(cid)
    return seen


def is_refusal(answer):
    return bool(REFUSAL_RE.search(answer or ""))


def rag_answer(question, call_llm, retrieve_fn=retrieve, k=3, tau=TAU, template=None, use_film_filter=False):
    """Trả lời một câu hỏi bằng một lượt truy xuất. ``call_llm(prompt) -> str``.

    Kết quả là dict gồm câu trả lời và các trường kiểm chứng (chunk truy xuất, chunk được trích,
    citation không có trong context, chunk bị gắn cờ injection, có từ chối bởi guardrail hay không).
    """
    chunks = retrieve_fn(question, k=k, filters=film_filters(question) if use_film_filter else None)
    retrieved_ids = [c["chunk_id"] for c in chunks]
    flagged = [c["chunk_id"] for c in chunks if sanitize(c["text"])[1]]
    if not chunks or max(c["score"] for c in chunks) < tau:
        return {"answer": REFUSAL_TEXT, "retrieved_chunk_ids": retrieved_ids, "cited_chunk_ids": [],
                "invalid_citation_ids": [], "injection_flagged_chunk_ids": flagged,
                "refused_by_guard": True, "n_llm_calls": 0}
    answer = call_llm(build_prompt(question, chunks, template)).strip()
    cited = extract_citations(answer)
    return {"answer": answer, "retrieved_chunk_ids": retrieved_ids, "cited_chunk_ids": cited,
            "invalid_citation_ids": [c for c in cited if c not in retrieved_ids],
            "injection_flagged_chunk_ids": flagged, "refused_by_guard": False, "n_llm_calls": 1}
