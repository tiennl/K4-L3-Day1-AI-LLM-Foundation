# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Khi temperature thấp (0.0 và 0.5), mô hình có xu hướng trả về các câu trả lời rất giống nhau, an toàn và mang tính phổ quát. Khi temperature tăng lên (1.0 và 1.5), các câu trả lời trở nên đa dạng, sáng tạo và cụ thể hơn, cho thấy tính ngẫu nhiên và phong phú trong cách chọn từ và ý tưởng được nâng cao.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Tôi sẽ đặt temperature khoảng 0.0–0.3. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, chính xác và bám sát chính sách/thông tin sản phẩm của công ty — không cần sự sáng tạo, mà cần độ tin cậy cao và ít rủi ro "bịa" thông tin (hallucination). Temperature thấp giúp hai người dùng hỏi cùng một câu sẽ nhận được câu trả lời tương tự nhau, dễ kiểm soát chất lượng và dễ debug khi có khiếu nại.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Theo bảng giá trong code (`PRICING_PER_1K_TOKENS`), output của GPT-4o là $0.010/1K token so với $0.0006/1K của mini — tức đắt hơn **~16.7 lần** (input cũng cùng tỉ lệ ~16.7 lần: $0.0025 so với $0.00015). Với 10.000 người dùng × 3 lần/ngày × 350 token đầu ra = 10.500.000 token/ngày, riêng chi phí output đã là **~$105/ngày** cho GPT-4o so với **~$6.30/ngày** cho mini (chưa tính input). Nên dùng GPT-4o khi tác vụ đòi hỏi suy luận phức tạp, độ chính xác cao hoặc rủi ro sai sót tốn kém (ví dụ: tư vấn pháp lý/tài chính, sinh code phức tạp). Nên dùng mini cho các tác vụ đơn giản, lặp lại, khối lượng lớn như trả lời FAQ, phân loại/tóm tắt ngắn — nơi chất lượng của mini đã đủ dùng và chênh lệch chi phí ở quy mô lớn là rất đáng kể.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Với persona "giáo viên tiểu học", phản hồi ngắn hơn, dùng từ ngữ đơn giản và ví von gần gũi (ví dụ so sánh blockchain với "cuốn sổ ghi chép mà ai cũng có một bản giống nhau"), tránh thuật ngữ kỹ thuật. Với persona "chuyên gia tài chính", phản hồi dài hơn, dùng thuật ngữ chuyên môn (sổ cái phân tán, cơ chế đồng thuận, băm mật mã, phi tập trung hóa) và trình bày có cấu trúc, chuyên sâu hơn về ứng dụng/rủi ro. Cùng một câu hỏi nhưng system prompt định hình hoàn toàn giọng văn, độ dài, mức độ kỹ thuật và đối tượng ví dụ minh họa — cho thấy system prompt là công cụ mạnh để "lập trình" hành vi và văn phong của model mà không cần đổi câu hỏi của người dùng.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Tôi thử với một đoạn ~114 từ tiếng Việt: ước lượng `số từ / 0.75` cho ra **152 token**, còn `count_tokens` (tiktoken, encoding `o200k_base` của gpt-4o) đếm được **127 token** — chênh khoảng **16%** (ước lượng thô cao hơn số thực). Điều thú vị là khi thử lại với encoding cũ hơn `cl100k_base` (dùng cho GPT-3.5/GPT-4 gốc), cùng đoạn văn đó lại tốn tới **2.23 token/từ** so với chỉ **1.16 token/từ** của một đoạn tiếng Anh tương đương — gần gấp đôi. Lý do: bộ từ vựng của tokenizer được huấn luyện chủ yếu trên dữ liệu tiếng Anh, nên các từ tiếng Việt có dấu (ví dụ "Trí" bị tách thành "Tr" + "í") thường không có sẵn thành một token nguyên, phải chẻ nhỏ theo byte/ký tự — khiến tiếng Việt "tốn" token hơn. Các encoding mới hơn như `o200k_base` (gpt-4o) đã mở rộng từ vựng đa ngôn ngữ nên khoảng cách này thu hẹp đáng kể, gần bằng tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất trong các giao diện chat tương tác trực tiếp với người dùng, đặc biệt khi phản hồi dài — người dùng thấy chữ xuất hiện ngay lập tức thay vì phải chờ vài giây "màn hình trắng", giúp trải nghiệm cảm thấy nhanh và tự nhiên hơn dù tổng thời gian xử lý không đổi. Ngược lại, non-streaming phù hợp hơn khi kết quả cần được xử lý theo chương trình trước khi hiển thị hoặc dùng tiếp — ví dụ khi cần parse JSON có cấu trúc, gọi function/tool calling, hoặc chạy trong pipeline nền (batch job) không có người xem trực tiếp — vì lúc đó phải đợi phản hồi đầy đủ mới dùng được, streaming chỉ thêm phức tạp mà không có lợi ích UX nào.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại (ví dụ 0.1s → 0.2s → 0.4s → 0.8s...), giúp giãn tải request ra theo thời gian và cho server đủ thời gian hồi phục thay vì bị dội thêm request ngay lập tức. Nếu hàng nghìn client cùng dùng delay cố định giống nhau (luôn chờ đúng 1 giây), tất cả sẽ retry đồng loạt cùng một thời điểm, tạo ra hiện tượng "thundering herd" — một đợt sóng request khổng lồ ập vào server đang quá tải, khiến nó tiếp tục sập hoặc quá tải nặng hơn, và chu kỳ lặp lại. Backoff (đặc biệt khi kết hợp thêm random jitter) làm các lần retry của các client rải đều ra, tránh hiện tượng đồng bộ hóa này.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> Tôi chọn persona "trợ giảng thân thiện của khóa học AI". System prompt: *"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, ưu tiên ví dụ thực tế thay vì lý thuyết dài dòng."* Tôi yêu cầu "trả lời ngắn gọn" vì học viên dùng chatbot để tra cứu nhanh giữa lúc thực hành, câu trả lời dài dòng làm mất mạch làm bài và tốn token/chi phí không cần thiết. Tôi chỉ định rõ "bằng tiếng Việt" vì học viên trong lớp chủ yếu dùng tiếng Việt, và nếu không ràng buộc, model có thể trả lời bằng tiếng Anh khi câu hỏi có lẫn thuật ngữ kỹ thuật tiếng Anh — gây khó hiểu cho người mới.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất là `run_assistant` chỉ giữ 3 lượt hội thoại gần nhất (`history[-6:]`) và không có bộ nhớ dài hạn — nếu học viên hỏi lại điều đã nói cách đây 4-5 lượt, trợ lý sẽ "quên" hoàn toàn. Cải thiện đề xuất: thêm bước tóm tắt (summarization) — mỗi khi history vượt quá ngưỡng, gọi model tóm tắt các lượt cũ thành 1-2 câu và lưu vào một biến `summary` riêng, sau đó ghép `summary` vào đầu system prompt cho các lượt tiếp theo thay vì xóa hẳn. Cách này giữ được ngữ cảnh quan trọng xuyên suốt phiên chat mà không làm phình to số token mỗi lần gọi API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
