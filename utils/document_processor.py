"""
Document processing module.
Handles PDF text extraction and text chunking.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from typing import List


class DocumentProcessor:
    """Handles document loading, text extraction, and chunking."""

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Extracted text as a string.
        """
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
        return text

    def split_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        source: str = "unknown"
    ) -> List[Document]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: The full text to split.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between consecutive chunks.
            source: Source document name for metadata.
            
        Returns:
            List of LangChain Document objects with metadata.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)

        return documents

    def get_document_stats(self, text: str) -> dict:
        """
        Get statistics about the extracted document.
        
        Args:
            text: The extracted text.
            
        Returns:
            Dictionary with document statistics.
        """
        words = text.split()
        return {
            "total_characters": len(text),
            "total_words": len(words),
            "total_pages": text.count("--- Page"),
            "avg_words_per_page": len(words) // max(text.count("--- Page"), 1)
        }
