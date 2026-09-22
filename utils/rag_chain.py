"""
RAG Chain module.
Handles vector store creation, retrieval, and LLM-powered Q&A.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List, Dict


# System prompt for the RAG chain
SYSTEM_PROMPT = """You are DocChat AI, an intelligent document assistant. Your job is to answer questions based solely on the provided context from the user's documents.

Rules:
1. Only answer based on the provided context. If the context doesn't contain enough information to answer the question, say "I couldn't find enough information in your documents to answer this question. Try rephrasing or uploading additional documents."
2. Be precise and cite which parts of the documents support your answer.
3. If the question is ambiguous, ask for clarification.
4. Format your responses clearly with proper structure when needed.
5. When quoting from the documents, use quotation marks.

Context from documents:
{context}

Answer the user's question based on the above context."""


class RAGChain:
    """Manages the RAG pipeline: embeddings, vector store, and query chain."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the RAG chain.
        
        Args:
            model_name: OpenAI model to use for generation.
        """
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a ChromaDB vector store from document chunks.
        
        Args:
            documents: List of LangChain Document objects to embed and store.
        """
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="docchat_collection"
        )

    def setup_chain(self, top_k: int = 4) -> None:
        """
        Set up the retrieval and generation chain.
        
        Args:
            top_k: Number of relevant chunks to retrieve per query.
        """
        if not self.vector_store:
            raise ValueError("Vector store not created. Call create_vector_store first.")

        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        # Create LLM
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.1,
            streaming=True
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        # Build the chain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            docs: List of retrieved documents.
            
        Returns:
            Formatted context string.
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "?")
            formatted.append(
                f"[Source: {source} | Chunk {chunk_id}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def query(self, question: str) -> Dict:
        """
        Query the RAG chain with a question.
        
        Args:
            question: The user's question.
            
        Returns:
            Dictionary with 'answer' and 'sources' keys.
        """
        if not self.chain:
            raise ValueError("Chain not set up. Call setup_chain first.")

        # Get relevant documents for source tracking
        relevant_docs = self.retriever.invoke(question)

        # Generate answer
        answer = self.chain.invoke(question)

        # Format sources
        sources = []
        for doc in relevant_docs:
            sources.append({
                "document": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "?"),
                "content": doc.page_content
            })

        return {
            "answer": answer,
            "sources": sources
        }

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        if not self.vector_store:
            return {"status": "No vector store created"}

        collection = self.vector_store._collection
        return {
            "total_documents": collection.count(),
            "collection_name": collection.name
        }
