from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Preserve the notebook defaults; both can be changed in the sidebar.
    chat_model: str = 'gemini-3.5-flash-lite'
    embedding_model: str = 'gemini-embedding-2-preview'
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks: int = 150

    def __post_init__(self):
        if not self.chat_model.strip() or not self.embedding_model.strip():
            raise ValueError('Tên model không được để trống.')
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError('Overlap phải nhỏ hơn chunk size và không âm.')
