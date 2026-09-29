"""Run: streamlit run app.py"""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import streamlit as st
from src.config import Settings
from src.pipeline import build_engine

st.set_page_config(page_title='GraphRAG PDF Assistant', page_icon='📚', layout='wide')


def reset_engine():
    engine = st.session_state.pop('engine', None)
    if engine is not None:
        engine.client.close()
    st.session_state.pop('inspector', None)


def forget_session():
    reset_engine()
    st.session_state['api_key'] = ''
    st.session_state['upload_version'] = st.session_state.get('upload_version', 0) + 1


st.title('📚 GraphRAG PDF Assistant')
st.caption('Hỏi đáp tài liệu với đồ thị tri thức, tìm kiếm kết hợp và bộ nhớ hội thoại.')
with st.sidebar:
    st.header('Cấu hình')
    api_key = st.text_input('Gemini API Key', type='password', key='api_key', on_change=reset_engine)
    st.caption('Key dùng trong phiên hiện tại. Văn bản PDF và câu hỏi được gửi tới Google Gemini khi xử lý; có thể phát sinh phí API.')
    with st.expander('Cấu hình nâng cao'):
        chat_model = st.text_input('Chat model', Settings.chat_model, on_change=reset_engine)
        embedding_model = st.text_input('Embedding model', Settings.embedding_model, on_change=reset_engine)
        chunk_size = st.number_input('Chunk size (ký tự)', 300, 5000, 1000, 100, on_change=reset_engine)
        overlap = st.number_input('Chunk overlap (ký tự)', 0, 1000, 200, 50, on_change=reset_engine)
    files = st.file_uploader('Chọn PDF', type=['pdf'], accept_multiple_files=True,
                            key=f"pdfs_{st.session_state.get('upload_version', 0)}")
    fingerprint = tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files)
    if fingerprint != st.session_state.get('files_fingerprint'):
        reset_engine()
        st.session_state['files_fingerprint'] = fingerprint
    process = st.button('Xử lý tài liệu', type='primary', disabled=not api_key.strip() or not files)
    st.button('Xóa phiên và API Key', on_click=forget_session)

if process:
    reset_engine()
    bar = st.progress(0, text='Đọc PDF…')
    try:
        settings = Settings(chat_model.strip(), embedding_model.strip(), int(chunk_size), int(overlap))
        with TemporaryDirectory(prefix='graphrag-') as folder:
            paths = []
            for index, uploaded in enumerate(files):
                path = Path(folder) / f'{index + 1}_{Path(uploaded.name).name}'
                path.write_bytes(uploaded.getvalue())
                paths.append(path)
            st.session_state.engine = build_engine(paths, api_key.strip(), settings,
                                                   lambda f, m: bar.progress(f, text=m))
        st.success('Tài liệu đã sẵn sàng. Bạn có thể đặt câu hỏi.')
    except ValueError as exc:
        st.error(str(exc).replace(api_key, '[REDACTED]'))
    except Exception as exc:
        import traceback
        st.error(f"Chi tiết lỗi: {exc}")
        st.code(traceback.format_exc())
    finally:
        bar.empty()

engine = st.session_state.get('engine')
if engine:
    a, b, c = st.columns(3)
    a.metric('Đoạn văn bản', len(engine.chunks))
    b.metric('Nút đồ thị', engine.graph.number_of_nodes())
    c.metric('Cạnh đồ thị', engine.graph.number_of_edges())
    if st.button("Xóa lịch sử trò chuyện"):
        if getattr(engine, "memory", None):
            engine.memory.clear(session_id="default")
        st.session_state.pop("inspector", None)
        st.rerun()

    history = (
        engine.memory.get_history(session_id="default")
        if getattr(engine, "memory", None)
        else []
    )
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info('Nhập API Key, tải PDF và bấm “Xử lý tài liệu” để bắt đầu.')

query = st.chat_input('Đặt câu hỏi về tài liệu…', disabled=engine is None)
if query and query.strip() and engine:
    try:
        with st.spinner('Đang tìm kiếm và tạo câu trả lời…'):
            _, debug = engine.chat(query.strip())
        st.session_state.inspector = debug
        st.rerun()
    except Exception:
        st.error('Chưa tạo được câu trả lời. Kiểm tra hạn mức API hoặc thử lại; lịch sử chưa được cập nhật.')
if st.session_state.get('inspector'):
    with st.expander('Thông tin truy xuất gần nhất'):
        st.json(st.session_state.inspector)
