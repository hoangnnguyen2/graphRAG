## Cài đặt và chạy

Yêu cầu Python 3.11 hoặc 3.12, kết nối internet và Gemini API Key có quyền dùng model đã chọn.

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell dùng: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Chạy các lệnh từ thư mục chứa `app.py`.

1. Nhập Gemini API Key vào sidebar.
2. Kiểm tra tên chat model và embedding model trong Cấu hình nâng cao.
3. Chọn một hoặc nhiều PDF có văn bản và bấm **Xử lý tài liệu**.
4. Đặt câu hỏi; xem thực thể, quan hệ, cộng đồng và các trang truy xuất ở phần thông tin truy xuất.
5. **Xóa lịch sử trò chuyện** giữ chỉ mục; **Xóa phiên và API Key** xóa cả engine, tài liệu đã chọn và key khỏi phiên ứng dụng.

Tên model mặc định được giữ từ notebook: `gemini-3.5-flash-lite` và `gemini-embedding-2-preview`. Chưa xác minh quyền truy cập/khả dụng bằng key thật. Có thể thay ở sidebar theo tài khoản; đổi model yêu cầu xử lý lại tài liệu.

## Cây thư mục

```text
graphrag-streamlit/
├── app.py                     # Giao diện Streamlit và trạng thái phiên
├── requirements.txt           # Cài package bằng pip
├── README.md
├── src/
│   └── graphrag/
│       ├── __init__.py
│       ├── config.py          # Cấu hình model, chunk và giới hạn
│       ├── schemas.py         # Schema Pydantic: entity, relationship
│       ├── preprocessing.py   # Đọc PDF, chia và làm sạch văn bản
│       ├── extraction.py      # Trích xuất tri thức bằng Gemini
│       ├── graph.py           # MultiDiGraph, Louvain communities 
|                              và tóm tắt thông tin cộng đồng
|       ├── memory.py          # Trí nhớ về các cuộc hội thoại trước đó
│       ├── graphRAG.py        # FAISS + BM25, graph retrieval
│       └── pipeline.py        # Pipeline

```
