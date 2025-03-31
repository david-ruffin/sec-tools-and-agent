from sec_api import ExtractorApi
import os
from dotenv import load_dotenv
from tools.logger import log_api_call, logger
import tempfile

# Load API key
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

@log_api_call
def sec_section_extractor(filing_url, section_code, return_type="text", original_query=None):
    """
    Extract a specific section from an SEC filing.
    
    Args:
        filing_url: URL to the SEC filing HTML document
        section_code: Section identifier (e.g., '1A' for Risk Factors, '7' for MD&A)
        return_type: Format of extraction ('text' or 'html')
        original_query: The original user query (unused, kept for compatibility)
    
    Returns:
        Dict with section data, metadata and status information
    """
    try:
        # Validate inputs
        if not filing_url:
            return {
                "status": "error",
                "message": "Filing URL is required",
                "data": None
            }
        
        if not section_code:
            return {
                "status": "error",
                "message": "Section code is required",
                "data": None
            }
        
        # Initialize the Extractor API
        extractor_api = ExtractorApi(api_key=SEC_API_KEY)
        
        # Extract the full section
        logger.info(f"Extracting section {section_code} from {filing_url}")
        full_content = extractor_api.get_section(filing_url, section_code, return_type)
        
        # Validate the response
        if not full_content or (isinstance(full_content, str) and len(full_content) < 50):
            return {
                "status": "no_results",
                "message": f"Section {section_code} not found or contains minimal content",
                "filing_url": filing_url,
                "section_code": section_code,
                "data": None
            }
        
        # Get content length for logging
        content_length = len(full_content)
        logger.info(f"Extracted content length: {content_length} characters")
        
        # Always save to temp file for RAG processing
        temp_file_path = None
        try:
            fd, temp_file_path = tempfile.mkstemp(prefix=f"sec_section_{section_code}_", suffix=".txt")
            with os.fdopen(fd, 'w') as temp_file:
                temp_file.write(full_content)
            logger.info(f"Saved full content to temp file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to save content to temp file: {str(e)}")
            
        # Always truncate for display but indicate it's truncated
        truncated_display = f"{full_content[:3000]}..."
        truncated_message = "Successfully extracted section"
        if content_length > 3000:
            truncated_message += f" {section_code} (truncated)"
            logger.info(f"Content truncated to 3000 characters (original: {content_length})")
        
        return {
            "status": "success", 
            "message": truncated_message,
            "data": truncated_display,
            "metadata": {
                "total_length": content_length,
                "section_code": section_code,
                "filing_url": filing_url,
                "needs_rag_processing": True,  # Always set to True
                "temp_file_path": temp_file_path
            }
        }
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error extracting section: {error_message}")
        return {
            "status": "error",
            "message": error_message,
            "error_type": str(type(e).__name__),
            "filing_url": filing_url if 'filing_url' in locals() else None,
            "section_code": section_code if 'section_code' in locals() else None,
            "data": None
        } 