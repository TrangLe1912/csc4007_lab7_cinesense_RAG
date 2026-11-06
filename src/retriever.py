"""Retriever của Lab 7: kho vector trong bộ nhớ + hàm ``retrieve(query, k, filters)``.

Giảng viên cung cấp sẵn, sinh viên không bắt buộc sửa.

* Mặc định dùng TF-IDF + cosine (chạy offline, không cần mạng) trên mini-KB CineSense v2.
* Sinh viên có thể thay bằng embedding của riêng mình qua ``Retriever(embedder=...)``;
  xem ``FunctionEmbedder`` bên dưới. Phần còn lại của pipeline (RAG, multi-hop) không đổi.
* Chữ ký ``retrieve(query, k, filters)`` giữ nguyên để nối thẳng sang Bài 9 (multi-hop).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parents[1]
DEFAULT_KB = BASE / "knowledge_base" / "kb_v2_full.jsonl"
REQUIRED_KEYS = ("chunk_id", "source_file", "doc_type", "film_id", "text")


def load_chunks(path=None):
    """Đọc JSONL, kiểm tra trường bắt buộc và ``chunk_id`` duy nhất."""
    path = Path(path or DEFAULT_KB)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in records:
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise ValueError(f"Chunk {r.get('chunk_id')} thiếu trường: {missing}")
    ids = [r["chunk_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Có chunk_id bị trùng")
    return records


class TfidfEmbedder:
    """Embedding thưa kiểu TF-IDF. Token giữ dấu gạch dưới nên ``film_beta`` là một token."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")

    def embed_documents(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def embed_query(self, query):
        return self.vectorizer.transform([query]).toarray()[0]


class FunctionEmbedder:
    """Bọc một hàm ``fn(list[str]) -> array (n, d)`` (ví dụ model embedding của Lab 7) thành embedder."""

    def __init__(self, fn):
        self.fn = fn

    def embed_documents(self, texts):
        return np.asarray(self.fn(list(texts)), dtype=float)

    def embed_query(self, query):
        return np.asarray(self.fn([query]), dtype=float)[0]


def _unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Retriever:
    def __init__(self, chunks=None, embedder=None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.embedder = embedder if embedder is not None else TfidfEmbedder()
        self.matrix = _unit_rows(self.embedder.embed_documents([c["text"] for c in self.chunks]))

    def search(self, query, k=5, filters=None):
        """Trả về tối đa ``k`` chunk, mỗi chunk thêm khoá ``score`` (cosine).

        ``filters`` là dict ``{trường: giá trị | [giá trị, ...]}`` trên metadata, ví dụ
        ``{"film_id": "film_gamma", "doc_type": "review"}``; lọc trước khi lấy top-k.
        Khi điểm bằng nhau, chunk đứng trước trong file được xếp trước (ổn định, tái lập được).
        Lưu ý: thứ hạng của các chunk có điểm 0 không mang ý nghĩa, chỉ biết chúng nằm cuối bảng.
        """
        if k < 1:
            raise ValueError("k phải >= 1")
        q = np.asarray(self.embedder.embed_query(query), dtype=float)
        norm = np.linalg.norm(q)
        scores = self.matrix @ (q / norm if norm else q)
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        out = []
        for i in order:
            chunk = self.chunks[i]
            if filters and any(chunk.get(key) not in (val if isinstance(val, (list, tuple, set)) else [val])
                               for key, val in filters.items()):
                continue
            out.append(dict(chunk, score=float(scores[i])))
            if len(out) == k:
                break
        return out


_DEFAULT = None


def retrieve(query, k=5, filters=None):
    """Hàm dùng chung cho RAG (Bài 8) và multi-hop (Bài 9); tạo retriever mặc định ở lần gọi đầu."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT.search(query, k=k, filters=filters)


def make_context(results):
    return "\n\n".join(f"[{r['chunk_id']}] (source: {r['source_file']}) {r['text']}" for r in results)
