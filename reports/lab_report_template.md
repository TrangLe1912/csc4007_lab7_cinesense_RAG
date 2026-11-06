# Lab 7 — Báo cáo CineSense RAG Assistant và nối sang Multi-hop

**Nhóm:** ...  **Ngày:** ...  **Mode:** API / Web UI / Sample-output  **Model hoặc Sample ID:** ...

## 1. Mục tiêu và giả thuyết
...

## 2. Dữ liệu, provenance, cấu hình chạy
KB v2 (40 chunk), `top_k`, có dùng `--film-filter` hay không, ngưỡng `TAU`, retriever (TF-IDF hay embedding riêng).

## 3. Truy xuất: k=1 vs k=3 vs lọc metadata

| Cấu hình | Hit@k (số câu có gold) | Câu bị miss | Nguyên nhân |
|---|---:|---|---|
| k=1 | | | |
| k=3 | | | |
| k=3 + film_id | | | |

Gold chỉ tính trên câu đã được gán nhãn; không khẳng định kết quả của toàn bộ tập.

## 4. RAG prompt và guardrails
Các quy tắc em đã thêm vào `rag_prompt.txt` và lý do. Lớp nào trong code đã chặn lỗi nào.

## 5. Citation, no-evidence, injection
Nêu ít nhất 2 câu evidence, 1 ca no-evidence, 1 ca injection. Kiểm citation ở mức *hỗ trợ thực tế*, không chỉ có mặt. Phân loại ít nhất 1 lỗi có căn cứ.

## 6. Nối sang multi-hop
- Q13, Q14: thực thể cầu nối, chuỗi chunk cần có, vì sao một lần truy xuất chưa đủ.
- Bộ test (>= 10 câu, >= 3 loại, >= 2 câu không trả lời trọn vẹn): tóm tắt.
- Bảng so sánh single-hop và multi-hop (4 tầng) và nhận xét: cải thiện ở loại câu nào, chi phí tăng bao nhiêu.

## 7. Phân tích lỗi multi-hop
Hop đầu tiên sai, error bucket (M1–M8), bằng chứng trong trace, sửa đổi đề xuất (mỗi lần đổi một thứ).

## 8. Kết luận và giới hạn
Đâu là kết quả tự chạy, đâu là output mẫu (`provided_sample_not_live_inference`). Hạn chế về cỡ mẫu và retriever.
