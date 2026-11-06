# CSC4007 — LAB 7: RAG ASSISTANT CÓ CITATION VÀ NỐI SANG MULTI-HOP

**Tên tiếng Việt:** Từ RAG một lượt có căn cứ đến câu hỏi cần nhiều bước  
**Dữ liệu:** CineSense KB v2 (film card + review) • **Thực hành trên lớp:** 135 phút (3 tiết, đề xuất)  
**Phương án:** Guided scaffold (cấp sẵn chunks, retriever, guardrails, khung multi-hop và độ đo)  
**Chế độ suy luận:** API, Web UI hoặc sample-output (phải khai báo đúng)  
**Phương thức nộp:** repository GitHub hoặc tệp nén theo hướng dẫn giảng viên.

## 1. Bối cảnh

Ở Lab 6, CineSense trả lời câu hỏi về review bằng một lượt truy xuất trên các review pack. Giờ kho tri thức mở rộng thêm **film card** (năm, đạo diễn, diễn viên, thể loại, thời lượng), và mỗi review mang thêm metadata `film_id` để nối *review → phim → thông tin phim*.

Trong lab này sinh viên làm ba việc liên tục trên cùng hệ thống:

- **Phần A – RAG Assistant (Bài 8):** cấu hình truy xuất, viết RAG prompt, trả lời kèm `chunk_id`, từ chối khi thiếu bằng chứng, không làm theo chỉ dẫn nằm trong dữ liệu.
- **Phần B – Kiểm chứng:** đo Hit@k, kiểm tra citation, phân loại lỗi retrieval và generation.
- **Phần C – Nối Bài 9:** thử các câu hỏi có thực thể cầu nối bị ẩn, xây bộ test multi-hop, hoàn thành vòng lặp Self-Ask và đánh giá 4 tầng.

**Câu hỏi cốt lõi:** Khi câu trả lời cần thông tin từ nhiều chunk, một lần truy xuất có đủ không? Và làm sao biết hệ thống dừng đúng lúc?

**Ngoài phạm vi:** fine-tuning, tự huấn luyện embedding, tự triển khai vector database, GraphRAG/HippoRAG, agent tự chủ.

## 2. Mục tiêu thực hành

Sau khi hoàn thành, sinh viên có thể: (i) cấu hình `k` và bộ lọc metadata `film_id` cho retriever; (ii) viết RAG prompt có citation, từ chối và kháng injection; (iii) đo Hit@k và kiểm tra citation, phân biệt lỗi retrieval với lỗi generation; (iv) giải thích vì sao một số câu hỏi cần nhiều hop và vì sao tăng `k` không thay thế được truy xuất lặp; (v) xây bộ test multi-hop có gold chain và chạy đánh giá theo 4 tầng (đáp án, bằng chứng, chuỗi hop, chi phí).

## 3. Học liệu được cấp

| Thành phần | Mô tả | Trách nhiệm sinh viên |
|---|---|---|
| Knowledge base | 40 chunk: 12 film card + 28 review, có `chunk_id`, `source_file`, `doc_type`, `film_id` | Kiểm tra nguồn, ID, metadata; không cần tự chunk |
| Retriever | Kho vector trong bộ nhớ, TF-IDF + cosine, `retrieve(query, k, filters)` | Thử `k` và bộ lọc `film_id`; tuỳ chọn thay bằng embedding của Lab trước |
| QA set | 8 evidence, 2 no_evidence, 2 injection, 2 multihop_preview | Kiểm tra retrieval, citation, refusal và bảo vệ chỉ dẫn |
| Guardrails trong code | Ngưỡng no-evidence, lọc injection, kiểm tra citation (`src/rag.py`) | Hiểu từng lớp, không phụ thuộc riêng vào prompt |
| Khung multi-hop | `src/multihop.py` với hai chỗ TODO; `src/eval_multihop.py` | Hoàn thành TODO-1 và TODO-2 |
| Bộ test mẫu | 5 câu theo lược đồ multi-hop | Xây bộ test của mình (>= 10 câu) |
| Sample outputs | Đầu ra dựng sẵn, tách khỏi live inference | Dùng để luyện đánh giá, không nói là do mình chạy LLM |

**Nguồn gốc dữ liệu:** Toàn bộ KB, QA và outputs mẫu là **synthetic teaching examples** (xem `data/DATA_PROVENANCE.md`). Không trộn với dữ liệu IMDB thật trong báo cáo nếu không ghi rõ.

## 4. Quy trình 135 phút

| Chặng | Thời gian | Việc phải làm | Minh chứng chính |
|---|---:|---|---|
| 1. Setup và data check | 15' | Chạy smoke test, đọc schema chunk, chọn mode, tạo manifest | `run_manifest.json`, console log |
| 2. Truy xuất và metadata | 25' | Thử `k=1/3`, thử `--film-filter`, ghi câu nào miss và vì sao | `retrieval_top*.csv` |
| 3. RAG prompt và guardrails | 35' | Hoàn thiện `prompts/rag_prompt.txt`, ghép context, sinh trả lời | `rag_prompt.txt`, `rag_answers.jsonl` |
| 4. Kiểm chứng | 25' | Hit@k, citation, no-evidence, injection, phân loại lỗi | `rag_checks.csv`, `lab_report.md` |
| 5. Nối multi-hop | 35' | Thử Q13–Q14, xây bộ test, hoàn thành TODO, chạy và chấm | `multihop_testset.jsonl`, `hop_trace_log.jsonl` |
| **Tổng** | **135'** | | |

### Chặng 1 – Kiểm tra môi trường và dữ liệu (15 phút)

1. Clone repo, cài `requirements.txt`, chạy `python scripts/smoke_test.py` và `python -m pytest -q tests`.
2. Đọc hai loại chunk (`doc_type = film_card` và `review`). Giải thích vì sao review không nhắc tên phim trong văn bản mà chỉ mang `film_id` trong metadata.
3. Tạo `run_manifest.json`: nhóm, thời điểm, mode, model hoặc sample ID, phiên bản dữ liệu, seed (nếu có), nhiệt độ (nếu kiểm soát được), `top_k`, có dùng `--film-filter` hay không.
4. Không lưu API key và không ghi tên `imdb_original` cho dữ liệu tự tạo.

### Chặng 2 – Truy xuất và metadata (25 phút)

1. Chạy `python scripts/run_retrieval.py --top-k 1`, rồi `--top-k 3`, rồi `--top-k 3 --film-filter`.
2. Với bốn câu có gold luyện tập (`data/qa_gold_practice.csv`), tính Hit@k cho từng cấu hình. Hit@k chỉ nói top-k có **ít nhất một** chunk vàng, không bảo đảm đủ bằng chứng.
3. Chọn một câu bị miss và giải thích bằng số liệu (điểm, chunk nhiễu xếp trên). Đề xuất một cách sửa (lọc metadata, viết lại truy vấn, đổi `k`).
4. Nếu dùng embedding riêng: cắm vào `Retriever(embedder=FunctionEmbedder(fn))` và **hiệu chỉnh lại ngưỡng** `TAU` trên các câu dev.

### Chặng 3 – RAG prompt và guardrails (35 phút)

1. Đọc `src/rag.py`: ba lớp bảo vệ nằm trong code (ngưỡng no-evidence, lọc injection, kiểm tra citation). Prompt là lớp thứ tư, không phải lớp duy nhất.
2. Hoàn thiện `prompts/rag_prompt.txt`: chỉ dùng context được cấp; citation bằng `chunk_id` có trong context và hỗ trợ đúng claim; từ chối khi thiếu bằng chứng; bỏ qua chỉ dẫn nằm trong tài liệu hoặc trong câu hỏi.
3. Chạy `python scripts/prepare_rag_contexts.py --top-k 3`, rồi sinh trả lời bằng API (`run_rag_api.py`), Web UI (`prepare_web_prompts.py`) hoặc nghiên cứu `samples/rag_answers_sample.jsonl`.
4. Lưu từng câu trả lời nguyên văn, context IDs và mode. Không bịa `chunk_id` khi câu trả lời không được hỗ trợ.

### Chặng 4 – Kiểm chứng (25 phút)

Bộ 14 QA gồm `8 evidence`, `2 no_evidence`, `2 injection`, `2 multihop_preview`:

- Chạy `python scripts/evaluate_rag.py --answers <file> --top-k 3` (thêm `--film-filter` nếu truy xuất có dùng).
- Evidence: đo Hit@k trên câu có gold; chấm thủ công `manual_claim_support`.
- No-evidence: hệ thống phải nói rõ không đủ căn cứ, không bịa nguồn.
- Injection: không làm theo lệnh trong câu hỏi hoặc trong chunk; phân biệt dữ liệu với chỉ dẫn điều khiển.
- Chọn ít nhất **1 lỗi có căn cứ** và phân loại: retrieval miss, `missing_citation`, `wrong_citation`, unsupported answer, guardrail failure. Nếu không có lỗi quan sát được, ghi rõ giới hạn cỡ mẫu, không tự tạo lỗi giả.

### Chặng 5 – Nối sang multi-hop (35 phút)

1. **Quan sát:** chạy single-hop trên Q13 và Q14 (`multihop_preview`). Với mỗi câu: thực thể cầu nối là gì, nó có trong câu hỏi không, chunk nào cần cho từng hop, top-k có đủ không?
2. **Bộ test:** viết `data/multihop_testset.jsonl` (lược đồ ở mục 5) rồi chạy `python scripts/validate_testset.py data/multihop_testset.jsonl`. Mỗi câu phải qua phép thử *bỏ một mắt xích vàng thì không trả lời được nữa*.
3. **Khung multi-hop:** hoàn thành `[TODO-1]` (chống lặp) và `[TODO-2]` (no-evidence theo hop) trong `src/multihop.py`.
4. **Chạy:** `python scripts/run_multihop.py --system multi` và `--system single`; nếu ở chế độ sample-output, chỉ phân tích `samples/hop_trace_sample.jsonl`.
5. **Chấm:** `python scripts/evaluate_multihop.py --testset data/multihop_testset.jsonl --runs "Single-hop=outputs/singlehop_runs.jsonl" "Multi-hop=outputs/hop_trace_log.jsonl"`.

**Về nhà / buổi lab sau:** hoàn thiện đánh giá 4 tầng, phân tích lỗi theo error buckets M1–M8 (tìm **hop đầu tiên bị sai**, mỗi lần chỉ đổi một thứ), kiểm thử ba guardrail (no-evidence theo hop, chunk injection, kiểm tra thực thể cầu nối).

## 5. Lược đồ bộ test multi-hop

Mỗi dòng của `multihop_testset.jsonl` là một JSON object:

| Trường | Ý nghĩa |
|---|---|
| `qid` | Mã câu hỏi, duy nhất |
| `question` | Câu hỏi (tiếng Anh, gửi cho retriever) |
| `type` | `bridge`, `comparison`, `bridge-comparison`, `intersection`, `implicit`, `single-hop` |
| `n_hops` | Số hop, bằng độ dài `gold_chain` |
| `answerable` | `full`, `partial` hoặc `none` |
| `gold_chain` | Danh sách `{hop, sub_question, gold_chunk_ids, intermediate_answer}`; hop không có bằng chứng thì `gold_chunk_ids = []` và `intermediate_answer = "NO_EVIDENCE"` |
| `gold_answer` | Đáp án ngắn; câu không trả lời được ghi `NO_EVIDENCE` |
| `gold_sp` | Các chunk bằng chứng, đúng bằng hợp các `gold_chunk_ids` trong `gold_chain` |

Yêu cầu tối thiểu: **>= 10 câu, >= 3 loại, >= 2 câu không trả lời được trọn vẹn.** Xem `data/multihop_testset_sample.jsonl` để lấy mẫu.

## 6. Ba chế độ chạy

| Chế độ | Thực hiện | Bằng chứng bắt buộc | Giới hạn |
|---|---|---|---|
| **A — API** | Chạy mô hình qua API do lớp cho phép; giữ phiên bản cấu hình | Prompt, input, raw output, model, run manifest, logs | Không push key; có thể không deterministic |
| **B — Web UI** | Dùng prompt đúng template trên giao diện chatbot | Prompt đã gửi, output chép nguyên văn, mô hình và thời điểm | Khó khóa version/model parameters; phải công bố. Multi-hop qua Web UI rất tốn công, nên chỉ thử tay vài câu |
| **C — Sample-output** | Dùng outputs minh họa; chạy eval/truy hồi offline | Các file được cấp, log tự chạy evaluator, nhận xét | **Không phải** minh chứng đã chạy LLM; không so sánh hiệu năng prompt do bản thân tạo |

## 7. Sản phẩm nộp

```text
submission/
├── README.md                          # tên nhóm, mode, cách chạy
├── run_manifest.json                  # cấu hình, provenance
├── prompts/
│   └── rag_prompt.txt
├── data/
│   └── multihop_testset.jsonl         # >= 10 câu, >= 3 loại, >= 2 câu không trả lời trọn vẹn
├── outputs/
│   ├── retrieval_top1.csv
│   ├── retrieval_top3.csv
│   ├── rag_answers.jsonl
│   ├── rag_checks.csv
│   ├── hop_trace_log.jsonl            # nhật ký từng hop
│   └── singlehop_vs_multihop.md       # so sánh theo 4 tầng độ đo
├── src/
│   └── multihop.py                    # đã hoàn thành TODO-1, TODO-2
└── reports/
    ├── multihop_error_analysis.md     # error buckets M1–M8 và guardrails đã kiểm thử
    └── lab_report.md
```

`lab_report.md` dài khoảng 2–3 trang theo `reports/lab_report_template.md`. Không bắt buộc phải có mọi file nếu API/Web UI bị giới hạn, nhưng phải ghi chính xác dữ liệu nào thực sự được chạy/chấm.

## 8. Thử thách chuyển giao (tuỳ chọn)

Chọn 6–8 bài tin tiếng Việt (VietNewsSense/UVN-1) có liên quan qua một thực thể chung. Viết 3 câu multi-hop (ít nhất 2 câu bridge) kèm gold chain. Xử lý đặc thù tiếng Việt: tách từ, chuẩn hoá biến thể tên riêng ("TP.HCM", "TP. Hồ Chí Minh"), có dấu và không dấu; cân nhắc truy xuất lai BM25 + embedding. So sánh hop coverage giữa CineSense và VietNewsSense, nêu 2 khó khăn riêng của tiếng Việt.
