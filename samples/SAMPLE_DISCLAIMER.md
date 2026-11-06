# Sample-output disclaimer

Các câu trả lời và nhật ký hop trong thư mục này được **dựng sẵn để thực hành đánh giá**, không do API, mô hình hay sinh viên tạo ra.

- `rag_answers_sample.jsonl`: một số câu cố ý sai (citation lệch, thiếu citation, bịa số liệu, làm theo chỉ dẫn độc hại, bỏ qua thực thể cầu nối) để người học phát hiện.
- `hop_trace_sample.jsonl`: hai nhật ký multi-hop (Q1 trả lời đầy đủ, Q6 dừng ở hop 2 vì không có bằng chứng) do khung `src/multihop.py` sinh ra với phần LLM được viết sẵn; `latency_s` để `null`.

Không báo cáo độ chính xác của các file này như kết quả LLM thật. Dùng chế độ sample-output thì phải ghi `provided_sample_not_live_inference` trong báo cáo.
