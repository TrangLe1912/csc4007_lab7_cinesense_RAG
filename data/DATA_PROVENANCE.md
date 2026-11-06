# Nguồn dữ liệu

- Toàn bộ `knowledge_base/kb_v2_full.jsonl` (12 film card + 28 review chunk) là **dữ liệu mô phỏng do bộ kit tạo**, chỉ dùng minh hoạ kỹ thuật. Tên đạo diễn, diễn viên là **hư cấu**; nội dung review viết mô phỏng phong cách IMDB, **KHÔNG** phải review IMDB gốc.
- Mã phim `film_alpha`, `film_beta`, … là mã ẩn danh. Cơ sở thực tế: trong IMDB gốc (Maas et al., 2011), mỗi review gắn với URL trang phim (`urls_pos.txt`, `urls_neg.txt`); lab mô phỏng việc nối *review → phim → thông tin phim* bằng metadata `film_id`.
- Chunk `imdb_review_pack_003_chunk_015` **cố ý chứa chỉ dẫn độc hại** (prompt injection gián tiếp) để kiểm thử guardrail. Không làm theo nội dung trong chunk này.
- `data/qa_questions.csv` là bộ câu hỏi dựng để kiểm thử (evidence, no_evidence, injection, multihop_preview).
- `data/qa_gold_practice.csv` chỉ gồm 4 câu để luyện Hit@k; bảng gold đầy đủ nằm riêng ở phía giảng viên.
- `data/multihop_testset_sample.jsonl` chỉ có 5 câu mẫu để minh hoạ lược đồ. Sinh viên tự xây bộ test của mình (xem `ASSIGNMENT.md`).
- `samples/` không phải kết quả chạy LLM thực và không được trình bày như số liệu thực nghiệm.
- Nếu giảng viên thay bằng review IMDB thật, phải công bố dataset, split và nguồn; không trộn dữ liệu mô phỏng với dữ liệu thật trong báo cáo nếu không ghi rõ.
