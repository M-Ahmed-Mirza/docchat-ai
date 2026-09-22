"""
DocChat AI — Chat with Your Documents
Main Streamlit application
"""

import streamlit as st
import os
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_processor import DocumentProcessor
from utils.rag_chain import RAGChain
from utils.config import get_config


def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if "uploaded_files_names" not in st.session_state:
        st.session_state.uploaded_files_names = []


def render_sidebar():
    """Render the sidebar with document upload and settings."""
    with st.sidebar:
        st.markdown("## 📄 Upload Documents")
        st.markdown("Upload PDF files to chat with their content.")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your OpenAI API key. Get one at platform.openai.com"
        )

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Model selection
        model = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Select the model for generating responses"
        )

        # Chunk settings
        st.markdown("### ⚙️ Retrieval Settings")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks"
        )
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of relevant chunks to retrieve"
        )

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files"
        )

        # Process documents button
        if uploaded_files:
            new_files = [f.name for f in uploaded_files]
            if st.button("🚀 Process Documents", use_container_width=True):
                if not api_key:
                    st.error("Please enter your OpenAI API key first.")
                    return None, None, None

                with st.spinner("Processing documents..."):
                    process_documents(
                        uploaded_files, chunk_size, chunk_overlap, top_k, model
                    )
                    st.session_state.uploaded_files_names = new_files
                    st.success(f"✅ {len(uploaded_files)} document(s) processed!")

        # Show loaded documents
        if st.session_state.uploaded_files_names:
            st.markdown("### 📚 Loaded Documents")
            for name in st.session_state.uploaded_files_names:
                st.markdown(f"- {name}")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Clear everything button
        if st.button("🔄 Reset Everything", use_container_width=True):
            st.session_state.messages = []
            st.session_state.rag_chain = None
            st.session_state.documents_loaded = False
            st.session_state.uploaded_files_names = []
            st.rerun()

        return api_key, model, top_k


def process_documents(uploaded_files, chunk_size, chunk_overlap, top_k, model):
    """Process uploaded PDF files and create RAG chain."""
    all_chunks = []
    processor = st.session_state.doc_processor

    for uploaded_file in uploaded_files:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # Extract text and create chunks
            text = processor.extract_text_from_pdf(tmp_path)
            chunks = processor.split_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source=uploaded_file.name
            )
            all_chunks.extend(chunks)
        finally:
            os.unlink(tmp_path)

    # Create RAG chain
    rag_chain = RAGChain(model_name=model)
    rag_chain.create_vector_store(all_chunks)
    rag_chain.setup_chain(top_k=top_k)

    st.session_state.rag_chain = rag_chain
    st.session_state.documents_loaded = True


def render_chat():
    """Render the chat interface."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message.get("sources"):
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** — {source['document']}")
                        st.markdown(f"> {source['content'][:300]}...")
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload and process documents first.")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.query(prompt)
                response = result["answer"]
                sources = result["sources"]

                st.markdown(response)

                if sources:
                    with st.expander("📖 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** — {source['document']}")
                            st.markdown(f"> {source['content'][:300]}...")
                            st.markdown("---")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="DocChat AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>📄 DocChat AI</h1>
            <p style='font-size: 1.2rem; color: #666;'>
                Upload documents. Ask questions. Get instant answers with sources.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Main content
    if not st.session_state.documents_loaded:
        # Welcome screen
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                ### 1️⃣ Upload
                Upload one or more PDF documents
                using the sidebar.
                """
            )
        with col2:
            st.markdown(
                """
                ### 2️⃣ Process
                Click 'Process Documents' to analyze
                and index your files.
                """
            )
        with col3:
            st.markdown(
                """
                ### 3️⃣ Chat
                Ask any question about your
                documents and get instant answers.
                """
            )

        st.markdown("---")
        st.info("👈 Start by uploading PDF documents in the sidebar.")
    else:
        render_chat()


if __name__ == "__main__":
    main()
