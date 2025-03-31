from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any, Dict, Optional
from tools.logger import logger

def rag_process(text_content: str, user_query: str, llm: Any) -> str:
    """
    Process text content using RAG approach.
    
    Args:
        text_content: The full text to process
        user_query: The original user question
        llm: LLM instance to use for answering
        
    Returns:
        A string containing the RAG-processed answer
    """
    # Skip RAG for empty or very short content
    if not text_content or len(text_content) < 500:
        logger.info("Text content too short for RAG processing, returning as is")
        return text_content
        
    # 1. Split text into chunks
    logger.info(f"Splitting text content into chunks (total length: {len(text_content)})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text_content)
    logger.info(f"Created {len(chunks)} chunks from text content")
    
    # 2. Create embeddings and vector store
    logger.info("Creating embeddings and vector store")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # 3. Set up retriever
    logger.info("Setting up retriever")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 4. Create and run QA chain
    logger.info("Creating and running QA chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # 5. Process query and return result
    enhanced_query = f"Using only the provided context, answer this question: {user_query}"
    logger.info(f"Processing query with RAG: '{enhanced_query}'")
    result = qa_chain({"query": enhanced_query})
    logger.info("RAG processing complete")
    return result["result"] 