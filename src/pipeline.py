from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.preprocessing import PDFPreprocessing
from src.extraction import Extraction
from src.graph import KnowledgeGraph
from src.graphRAG import GraphRAG
from src.memory import ChatMemoryManager  # <-- Thêm import

def build_engine(paths, api_key, settings, progress=lambda fraction, message: None):
    if not api_key.strip():
        raise ValueError('Vui lòng nhập Gemini API Key.')
    chunks = PDFPreprocessing(settings.chunk_size, settings.chunk_overlap).preprocessing(paths)
    if not chunks:
        raise ValueError('PDF không có văn bản đọc được. PDF scan cần OCR trước.')
    if len(chunks) > settings.max_chunks:
        raise ValueError(f'Tài liệu có {len(chunks)} đoạn, vượt giới hạn {settings.max_chunks}. Hãy chia nhỏ PDF.')
    client = genai.Client(api_key=api_key)
    try:
        knowledge_graph = KnowledgeGraph(chunks, api_key, settings.chat_model)
        graph = knowledge_graph.build_graph()
        progress(.72, 'Phân cụm và tóm tắt cộng đồng…')
        summaries = knowledge_graph.generate_summary_communities()
        progress(.9, 'Tạo chỉ mục FAISS và BM25…')
        embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=api_key)
        node2cid = knowledge_graph.get_node2cid()
        
        memory_manager = ChatMemoryManager(api_key, settings.chat_model)
        engine = GraphRAG(graph, node2cid, summaries, chunks, api_key, settings.chat_model, embeddings, memory_manager=memory_manager)
        
        progress(1.0, 'Hoàn tất')
        return engine
    except Exception:
        client.close()
        raise
