from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import uuid
from pathlib import Path

class PDFPreprocessing:
    def __init__(self, CHUNK_SIZE=1000, CHUNK_OVERLAP=200):
        self.chunk_size=CHUNK_SIZE
        self.chunk_overlap=CHUNK_OVERLAP
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    @staticmethod
    def _clean_text(text):
        cleaned_text=re.sub(r"\s+", " ", text)
        return cleaned_text.strip()
        
    @staticmethod
    def _convert_to_pdf_files(input_paths):
        pdf_paths = []
        
        if isinstance(input_paths, (list, tuple)):
            for p in input_paths:
                path = Path(p)
                if path.is_file() and path.suffix.lower() == ".pdf":
                    pdf_paths.append(path)
                elif path.is_dir():
                    pdf_paths.extend(list(path.glob("*.pdf")))
        else:
            path = Path(input_paths)
            if path.is_dir():
                pdf_paths = list(path.glob("*.pdf"))
            elif path.is_file() and path.suffix.lower() == ".pdf":
                pdf_paths = [path]
                    
        if not pdf_paths:
            raise FileNotFoundError("Không tìm thấy file .pdf nào từ đầu vào đã cung cấp.")

        return pdf_paths

    def load_pdf(self, pdf_files):
        raw_documents = []
        for file in pdf_files:
            try:
                loader = PyMuPDFLoader(str(file))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file.name
                raw_documents.extend(docs)

            except Exception as e:
                raise ValueError(f"Không đọc được PDF: {file.name}") from e

        return raw_documents


    def split_document(self, docs):
        chunks = self.splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["id"] = str(uuid.uuid4())
            chunk.page_content = self._clean_text(chunk.page_content)

        return chunks

    def preprocessing(self, folder_paths):
        pdf_paths = self._convert_to_pdf_files(folder_paths)
        documents = self.load_pdf(pdf_paths)
        chunks = self.split_document(documents)

        return chunks
