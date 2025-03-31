from tools.logger import logger
from tools.rag_handler import rag_process
import json
import tempfile
import os
import sys

def sec_rag_processor(text=None, query=None, source=None, file_path=None):
    """
    Process large text with RAG to extract relevant information for a specific query.
    
    Args:
        text: Large text content to process with RAG (either text or file_path must be provided)
        query: The user query to focus the extraction on
        source: Optional source information (e.g., 'section 1A of 10-K')
        file_path: Path to a file containing the text to process (alternative to passing text directly)
    
    Returns:
        Dict with processed text and status information
    """
    try:
        # Validate inputs
        if not query:
            return {
                "status": "error",
                "message": "Query is required for RAG processing",
                "data": "Error: No query provided for RAG processing"
            }
            
        # Get text content from file if provided
        content = text
        if not content and file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                logger.info(f"Read content from file: {file_path}")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error reading file: {str(e)}",
                    "data": f"Error: Could not read file {file_path}"
                }
                
        if not content:
            return {
                "status": "error",
                "message": "No content provided (either text or file_path must be provided)",
                "data": "Error: No content to process"
            }
            
        # Check if content is too short
        if len(content) < 500:
            return {
                "status": "skipped",
                "message": "Text too short for RAG processing",
                "original_length": len(content),
                "data": content,
                "source": source
            }
            
        # Import needed modules only when needed
        logger.info(f"Processing text with RAG (length: {len(content)})")
        
        # Import OpenAI directly to create an LLM if needed
        from langchain_openai import ChatOpenAI
        
        # Create a new LLM instance
        try:
            llm = ChatOpenAI(temperature=0)
            logger.info("Created new LLM instance for RAG processing")
            
            # Process through RAG
            processed_text = rag_process(content, query, llm)
            logger.info("Successfully processed with RAG")
            
            # Add processing note
            if source:
                processed_text += f"\n\n[This is a processed summary of {source} ({len(content)} characters)]"
            else:
                processed_text += f"\n\n[This is a processed summary of the original text ({len(content)} characters)]"
                
            return {
                "status": "success",
                "message": "Successfully processed with RAG",
                "original_length": len(content),
                "processed_length": len(processed_text),
                "data": processed_text,
                "source": source
            }
        except Exception as e:
            logger.error(f"Error creating LLM or processing with RAG: {str(e)}")
            truncated = content[:3000] + f"\n\n[Content truncated due to length ({len(content)} characters)]" if len(content) > 3000 else content
            return {
                "status": "error",
                "message": f"Error in RAG processing: {str(e)}",
                "original_length": len(content),
                "data": truncated,
                "source": source
            }
            
    except Exception as e:
        logger.error(f"Error in RAG processing: {str(e)}")
        # If RAG fails, truncate to avoid context window issues
        truncated = ""
        if text and len(text) > 0:
            truncated = text[:3000] + f"\n\n[Content truncated due to length ({len(text)} characters)]" if len(text) > 3000 else text
        
        return {
            "status": "error",
            "message": f"Error in RAG processing: {str(e)}",
            "error_type": str(type(e).__name__),
            "data": truncated,
            "source": source
        }

def save_text_to_temp_file(text):
    """
    Save text to a temporary file and return the file path.
    Useful for handling large text that would exceed context limits.
    """
    try:
        fd, path = tempfile.mkstemp(suffix='.txt', prefix='sec_rag_')
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        return path
    except Exception as e:
        logger.error(f"Error saving text to temp file: {str(e)}")
        return None 