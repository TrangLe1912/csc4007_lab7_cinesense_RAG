# Rubric Lab 7 (thang 10 điểm) – Phần RAG Assistant và nối multi-hop

Đây là **rubric hoạt động lab độc lập**, chưa ánh xạ OBE hoặc thay thế bất kỳ rubric chính thức nào của học phần.

| Hạng mục | Điểm | Đạt yêu cầu khi |
|---|---:|---|
| Cấu hình truy xuất | 2.0 | Chạy `k=1/3` và có bộ lọc `film_id`; Hit@k trên câu có gold; giải thích được ít nhất 1 ca miss bằng số liệu |
| RAG prompt và guardrails | 2.0 | Prompt có đủ 4 quy tắc (chỉ dùng context, citation, từ chối, kháng injection); câu trả lời gắn `chunk_id` có trong context |
| Kiểm chứng citation, no-evidence, injection | 2.0 | Kiểm thử đúng loại ca, có chấm support thủ công, phân loại đúng ít nhất 1 lỗi có căn cứ |
| Nối multi-hop | 2.5 | Giải thích đúng thực thể cầu nối của Q13–Q14; bộ test qua `validate_testset.py`; hoàn thành hai TODO; có trace và bảng so sánh phù hợp mode |
| Báo cáo, tái hiện và minh bạch nguồn | 1.5 | Manifest, báo cáo, file output phù hợp mode, phân biệt đầu ra mẫu/live, không lộ credentials |
| **Tổng** | **10.0** | |

## Thang mô tả gợi ý

- **Mức đầy đủ (>=80% của từng tiêu chí):** hoạt động đúng, có minh chứng rõ, giải thích nguyên nhân/giới hạn.
- **Mức cơ bản (50–<80%):** chạy được nhiệm vụ chính nhưng thiếu phân tích hoặc một phần minh chứng.
- **Mức chưa đạt (<50%):** không có bằng chứng, không thực hiện hoặc kết luận không gắn dữ liệu.

## Phần sản phẩm học tập của Bài 9

Bốn sản phẩm của Bài 9 (bộ test multi-hop, nhật ký hop, bảng so sánh single/multi, phân tích lỗi) có thể được chấm riêng theo rubric của Bài 9 (Phụ lục D của kịch bản): bộ câu hỏi và gold chain 20%, vòng lặp multi-hop 25%, đánh giá 4 tầng 25%, phân tích lỗi 20%, guardrails 10%. Giảng viên chọn dùng rubric nào cho phần nối multi-hop, không cộng dồn hai lần.

### Nguyên tắc công bằng giữa ba chế độ

API/Web UI có thể cung cấp **bằng chứng chạy mô hình**. Sample-output chỉ có thể chứng minh **khả năng đọc, đánh giá và phân tích đầu ra được cấp**, không thể được mô tả là người học đã chạy LLM. Giảng viên chấm phần đánh giá/kỹ thuật thực sự làm được và yêu cầu khai báo giới hạn, không tự giả định mode C có bằng chứng inference tương đương. Phần xây bộ test và hoàn thành TODO làm được ở cả ba chế độ.
