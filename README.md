# CSC4007 – Lab 7: RAG Assistant có citation & nối sang Multi-hop (Guided Scaffold)

**Thời lượng đề xuất:** 135 phút • **Ngữ cảnh:** CineSense/IMDB • **Ba chế độ:** API / Web UI / đầu ra mẫu.

> Bản đi kèm có ngữ liệu **GIẢ LẬP CHỈ ĐỂ DẠY/SMOKE TEST**: tên đạo diễn, diễn viên là hư cấu, review viết mô phỏng phong cách IMDB, đầu ra mẫu không phải kết quả suy luận thật của LLM. Chunk `imdb_review_pack_003_chunk_015` **cố ý chứa chỉ dẫn độc hại** để kiểm thử guardrail.

Lab 7 nối tiếp Lab 6 (Prompt Engineering + RAG TF-IDF). Lab này đi tiếp ba bước:

1. **RAG Assistant (Bài 7):** kho tri thức có *film card* và metadata `film_id`, hàm `retrieve(query, k, filters)`, trả lời kèm citation, từ chối khi thiếu bằng chứng, chống prompt injection.
2. **Kiểm chứng:** Hit@k, citation, no-evidence, injection, phân loại lỗi retrieval và generation.
3. **Nối Bài 8 (Multi-hop Reasoning):** thử câu hỏi cần *thực thể cầu nối*, rồi dùng khung `src/multihop.py` (hai chỗ TODO) và bộ độ đo 4 tầng.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_test.py
python -m pytest -q tests
```

## Đường chạy cơ bản (offline, không gọi LLM)

```bash
python scripts/run_retrieval.py --top-k 1
python scripts/run_retrieval.py --top-k 3
python scripts/run_retrieval.py --top-k 3 --film-filter
python scripts/prepare_rag_contexts.py --top-k 3
python scripts/evaluate_rag.py --answers samples/rag_answers_sample.jsonl --top-k 3
python scripts/validate_testset.py data/multihop_testset_sample.jsonl --min-questions 5 --min-unanswerable 1
python scripts/evaluate_multihop.py --testset data/multihop_testset_sample.jsonl --runs "Multi-hop (mẫu)=samples/hop_trace_sample.jsonl"
```

Các lệnh trên chạy **offline** trên dữ liệu và đầu ra mẫu. Đây là đường kiểm thử pipeline, **không phải bằng chứng đã gọi LLM**.

## Cách làm 3 chế độ

- **API:** thiết lập **biến môi trường** theo `.env.example` (file `.env` không được tự động nạp; trên macOS/Linux có thể chạy `set -a; source .env; set +a`), rồi chạy `python scripts/run_rag_api.py --top-k 3`. `LLM_API_URL` phải là URL endpoint OpenAI-compatible `/chat/completions` mà người học được phép sử dụng. Không đưa API key lên repo. Phần multi-hop: `python scripts/run_multihop.py --system multi` (sau khi hoàn thành hai TODO) và `--system single` để lấy baseline.
- **Web UI:** chạy `python scripts/prepare_web_prompts.py`, mở `outputs/web_prompts/`, gửi lần lượt vào giao diện; lưu nguyên văn phản hồi vào JSONL đúng schema như trong `samples/`. Không tự động hóa website bằng script này.
- **Sample-output:** dùng `samples/*` để chấm và phân tích; đánh dấu nguồn `provided_sample_not_live_inference`. Sinh viên **vẫn tự viết RAG prompt**, tự xây bộ test multi-hop và phân tích giới hạn, nhưng không được tuyên bố đã chạy inference mới.

Xem `ASSIGNMENT.md`, `RUBRIC.md` và `reports/lab_report_template.md`.

## Cấu trúc

- `data/`: 14 câu hỏi QA (8 evidence, 2 no_evidence, 2 injection, 2 multihop_preview), gold luyện tập cho 4 câu, 5 câu multi-hop mẫu.
- `knowledge_base/`: `kb_v2_full.jsonl` gồm 40 chunk (12 film card + 28 review), mỗi chunk có `chunk_id`, `source_file`, `doc_type`, `film_id`.
- `src/retriever.py`: kho vector trong bộ nhớ, mặc định TF-IDF + cosine; thay embedding bằng `Retriever(embedder=FunctionEmbedder(fn))`.
- `src/rag.py`: RAG một lượt với ba lớp guardrail (no-evidence theo ngưỡng, lọc injection, kiểm tra citation).
- `src/multihop.py`: vòng lặp multi-hop kiểu Self-Ask, **sinh viên hoàn thành TODO-1 và TODO-2** (Bài 9).
- `src/eval_multihop.py`, `src/evaluation.py`: độ đo 4 tầng, Hit@k, kiểm tra citation.
- `prompts/rag_prompt.txt`: prompt người học hoàn thiện.
- `samples/`: đầu ra dựng sẵn để luyện đánh giá, không phải output API thật.
- `scripts/`: kiểm thử dữ liệu, truy hồi, chuẩn bị context, chạy API, chấm RAG, kiểm tra bộ test, chạy và chấm multi-hop.
- `outputs/`: kết quả được tạo khi chạy; không nộp log giả làm thực nghiệm.

**Lưu ý:** Gold QA đầy đủ và hướng dẫn giảng viên được phát riêng, không nằm trong gói student. Điểm similarity phụ thuộc retriever: với embedding của bạn, con số sẽ khác (ngưỡng `TAU = 0.15` chỉ dành cho TF-IDF demo). Thứ hạng của các chunk có điểm 0 không mang ý nghĩa.

## Lịch sử phiên bản

Repo hiện được dùng cho Lab 7 RAG Assistant nối sang Multi-hop. Bản Lab 7 cũ (scaffold prompt tháng 4/2026) được giữ ở nhánh [`archive/lab7-prompt-scaffold-before-rag-2026-11-06`](https://github.com/TrangLe1912/csc4007_lab7_cinesense_RAG/tree/archive/lab7-prompt-scaffold-before-rag-2026-11-06). Xem `VERSION_HISTORY.md`.
