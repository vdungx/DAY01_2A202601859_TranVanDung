# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng câu trả lời mặc định bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, câu trả lời có xu hướng giảm tính nhất quán (deterministic) và tăng tính sáng tạo (creativity). Ở mức 0.0, phản hồi lặp lại chính xác cùng nội dung và từ ngữ chuẩn mực. Ở mức 1.0, cấu trúc câu bắt đầu phong phú và đa dạng hơn về góc nhìn. Khi lên tới 1.5, văn bản bắt đầu rời rạc, lủng củng và xuất hiện các cụm từ ít phổ biến do mô hình chọn các token có xác suất rất thấp.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature từ 0.0 đến 0.2 cho chatbot hỗ trợ khách hàng. Đối với nghiệp vụ chăm sóc khách hàng, yêu cầu hàng đầu là sự chính xác, nhất quán và độ tin cậy của thông tin (không tự ý bịa đặt hay ảo tưởng thông tin). Mức nhiệt độ thấp giúp kiểm soát chặt chẽ nội dung đầu ra dựa trên tài liệu nghiệp vụ đã có.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 16.67 lần cho cả input ($2.5 vs $0.15) và output ($10.0 vs $0.6). Đối với workload 10.000 users/ngày x 3 lần x 350 tokens, GPT-4o sẽ tốn chi phí cao hơn rõ rệt. GPT-4o xứng đáng khi cần giải quyết các tác vụ tư duy logic phức tạp, viết code giải thuật khó hoặc phân tích tài chính chuyên sâu. Trong khi đó, GPT-4o-mini phù hợp cho các cuộc hội thoại thông thường, tóm tắt văn bản ngắn hoặc phân loại ý định khách hàng (intent classification).

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi của "giáo viên tiểu học" rất ngắn gọn, dùng từ ngữ siêu đơn giản và so sánh blockchain với "cuốn sổ cái chung của cả lớp mà ai cũng nhìn thấy". Ngược lại, phản hồi của "chuyên gia tài chính" dài hơn, cấu trúc chặt chẽ và sử dụng nhiều thuật ngữ như "sổ cái phi tập trung", "mật mã học", "hợp đồng thông minh". System prompt đã thay đổi phân phối xác suất sinh token của mô hình, hướng nó chọn các từ vựng, văn phong và ví dụ phù hợp nhất với vai trò được định nghĩa.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, tiktoken thực tế đếm được khoảng 160-180 tokens, trong khi công thức ước lượng thô chỉ cho ra 133 tokens (chênh lệch khoảng 20% - 35%). Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì các Tokenizer của OpenAI tối ưu hóa cho tiếng Anh (một từ thông dụng là 1 token), trong khi tiếng Việt là ngôn ngữ ít phổ biến hơn và có hệ thống dấu phức tạp, khiến các từ đơn hoặc âm tiết bị phân tách thành các sub-word hoặc byte nhỏ hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng trò chuyện trực tiếp (Chatbot) hoặc khi sinh các văn bản dài (như viết blog, code) vì nó làm giảm đáng kể thời gian phản hồi đầu tiên (Time to First Token), giúp người dùng cảm thấy ứng dụng hoạt động ngay lập tức thay vì phải chờ đợi. Ngược lại, non-streaming phù hợp hơn cho các tác vụ chạy ngầm (background jobs), xử lý dữ liệu hàng loạt (batch processing), hoặc khi hệ thống cần kiểm duyệt nội dung, dịch thuật và định dạng lại dữ liệu thô (ví dụ: parse JSON) trước khi hiển thị cho người dùng cuối.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giãn cách thời gian giữa các lần thử lại sau mỗi lần lỗi, từ đó giảm đáng kể áp lực dồn dập lên hệ thống đang bị quá tải, tạo cơ hội cho máy chủ tự phục hồi. Nếu hàng nghìn client cùng sử dụng delay cố định 1 giây, họ sẽ tạo ra hiệu ứng "Thảm họa Thử lại" (Thundering Herd / Retry Storm). Khi đó, các client sẽ đồng loạt gửi yêu cầu vào cùng một thời điểm sau mỗi giây, khiến máy chủ liên tục bị quá tải theo chu kỳ và không thể phục hồi được trạng thái hoạt động bình thường.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona "Trợ lý học tập Python & AI thân thiện, trả lời ngắn gọn". System prompt: "Bạn là Mentor AI thân thiện, giải đáp các thắc mắc về lập trình Python và trí tuệ nhân tạo. Hãy phản hồi ngắn gọn, súc tích trong 3-5 câu bằng tiếng Việt tự nhiên và cung cấp ví dụ code ngắn nếu cần thiết." Việc chỉ định "trợ giảng thân thiện" giúp tạo giọng điệu tích cực, khuyến khích học viên; yêu cầu "ngắn gọn, súc tích trong 3-5 câu" giúp tránh việc câu trả lời quá dài làm tràn màn hình giao diện dòng lệnh CLI.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là dung lượng bộ nhớ lịch sử cực kỳ ngắn (chỉ lưu tối đa 3 lượt chat gần nhất do câu lệnh cắt history[-6:]). Đề xuất cải thiện: Tích hợp cơ chế "Tóm tắt lịch sử tự động" (Conversation Summary Memory). Cách triển khai: Khi số lượng tin nhắn vượt quá 6, ta gọi một model mini để tóm tắt các tin nhắn cũ hơn thành một đoạn tóm tắt ngắn (summary context), sau đó lưu đoạn tóm tắt này và chèn vào đầu các tin nhắn mới gửi lên API (ngay sau system prompt) để mô hình vẫn hiểu được bối cảnh hội thoại từ đầu.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
