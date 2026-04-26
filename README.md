# Lab 7 CineSense Prompt Scaffold

## Mục đích

Bộ này là **lab scaffold tối thiểu**, không phải starter kit hoàn chỉnh.

Sinh viên cần tự thiết kế prompt, chạy thử nghiệm, phân tích lỗi và cải tiến prompt trên ngữ cảnh IMDB Movie Review Dataset đã được dùng xuyên suốt từ Bài 2 đến Bài 6.

Lab 7 nối tiếp Bài 6 về Transformer/self-attention/fine-tuning bằng cách đặt câu hỏi thực tế:

> Khi dùng LLM cho tác vụ phân tích review phim, làm sao biết prompt có đáng tin cậy hay không?

## Cấu trúc thư mục

```text
lab7_cinesense_prompt_scaffold/
├── data/
│   ├── imdb_sample_50.csv
│   └── student_testset_template.csv
├── prompts/
│   ├── prompt_template_v1.txt
│   ├── prompt_template_v2.txt
│   └── system_prompt_optional.txt
├── eval/
│   ├── eval_template.csv
│   └── error_bucket_guide.md
├── outputs_sample/
│   ├── result_v1_sample.csv
│   └── result_v2_sample.csv
├── scripts/
│   └── run_prompt_eval_skeleton.py
└── submissions/
    └── error_analysis_template.md
```

## Dữ liệu mẫu

File `data/imdb_sample_50.csv` gồm 50 review ngắn theo phong cách IMDB, được tạo để minh họa cho lab. Đây không phải dữ liệu gốc của IMDB. Khi triển khai chính thức, giảng viên hoặc sinh viên có thể thay thế bằng một mẫu lấy từ IMDB dataset đang dùng trong các bài trước.

Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| review_id | Mã review |
| review_text | Nội dung review |
| gold_sentiment | Nhãn chuẩn: positive hoặc negative |
| expected_aspects | Một số khía cạnh đáng chú ý |
| difficulty | Mức độ dễ/khó của mẫu |
| source_note | Ghi chú nguồn |

## Cách dùng gợi ý

### Bước 1: Chọn tác vụ

Tối thiểu chọn tác vụ:

> Phân loại sentiment của review phim: positive/negative.

Nhóm khá có thể mở rộng thêm:

- trích xuất aspect;
- giải thích bằng chứng;
- đánh giá confidence;
- phát hiện hallucination.

### Bước 2: Viết Prompt v1

Mở file:

```text
prompts/prompt_template_v1.txt
```

Có thể dùng ngay hoặc sửa lại.

### Bước 3: Chạy trên 20–50 review

Có ba cách:

1. Dùng ChatGPT/Gemini/Claude qua web UI;
2. Dùng API;
3. Dùng file output mẫu nếu lỗi mạng/API.

Không bắt buộc phải dùng API nếu mục tiêu buổi học là đánh giá prompt.

### Bước 4: Đánh giá lỗi

Dùng:

```text
eval/eval_template.csv
eval/error_bucket_guide.md
```

Mỗi dòng cần xác định:

- sentiment đúng hay sai;
- JSON có hợp lệ không;
- evidence có lấy đúng từ review không;
- có hallucination không;
- lỗi thuộc bucket nào.

### Bước 5: Cải tiến Prompt v2

Mở:

```text
prompts/prompt_template_v2.txt
```

Sau đó sửa prompt dựa trên lỗi của Prompt v1.

### Bước 6: Viết báo cáo

Dùng:

```text
submissions/error_analysis_template.md
```

## Yêu cầu nộp tối thiểu

```text
lab7_submission/
├── prompt_v1.txt
├── prompt_v2.txt
├── testset.csv
├── result_v1.csv
├── result_v2.csv
├── eval_v1.csv
├── eval_v2.csv
└── error_analysis.md
```

## Tiêu chí đánh giá gợi ý

| Tiêu chí | Điểm |
|---|---:|
| Có testset 20–50 mẫu rõ ràng | 2 |
| Prompt v1 và v2 có khác biệt có chủ đích | 2 |
| Có kết quả chạy và bảng đánh giá | 2 |
| Có error buckets và ví dụ cụ thể | 2 |
| Nhận xét cải tiến hợp lý | 2 |

## Lưu ý quan trọng

Không đánh giá prompt bằng cảm giác. Cần có bằng chứng qua testset, bảng lỗi và ví dụ cụ thể.
