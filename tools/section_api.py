from sec_api import ExtractorApi
import os
from dotenv import load_dotenv
from tools.logger import log_api_call

# Load API key
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

@log_api_call
def sec_section_extractor(filing_url, section_code, return_type="text", mode="auto"):
    """
    Extract a specific section from an SEC filing.
    
    Args:
        filing_url: URL to the SEC filing HTML document
        section_code: Section identifier (e.g., '1A' for Risk Factors, '7' for MD&A)
        return_type: Format of extraction ('text' or 'html')
        mode: How to handle large sections ('auto', 'summary', 'full', 'outline')
    
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
        
        # Extract the section
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
        
        # Get content length
        content_length = len(full_content)
        
        # Process based on mode and size
        if content_length > 6000 and mode == "auto":
            # For large content in auto mode, return outline and first chunk
            
            # Simple paragraph chunking - find natural breaks
            paragraphs = [p for p in full_content.split('\n\n') if p.strip()]
            
            # Get the first few paragraphs (introduction/overview)
            intro = '\n\n'.join(paragraphs[:3])
            
            # Create simple metadata
            metadata = {
                "total_length": content_length,
                "paragraph_count": len(paragraphs),
                "section_code": section_code,
                "filing_url": filing_url
            }
            
            return {
                "status": "success",
                "message": f"Large section split into chunks. Returning intro and metadata.",
                "data": intro,
                "metadata": metadata,
                "has_more": True
            }
            
        elif mode == "summary":
            # Return just a summary (first few paragraphs and length info)
            paragraphs = [p for p in full_content.split('\n\n') if p.strip()]
            summary = '\n\n'.join(paragraphs[:5])  # First 5 paragraphs as summary
            
            return {
                "status": "success",
                "message": f"Returning summary of section {section_code}",
                "data": summary,
                "total_length": content_length,
                "has_more": content_length > len(summary)
            }
            
        elif mode == "outline":
            # Extract headings using simple text analysis
            lines = full_content.split('\n')
            # Heuristic: Headings are often shorter lines ending with a colon or all caps
            potential_headings = [line for line in lines if 
                                  len(line.strip()) < 100 and 
                                  (line.strip().isupper() or 
                                   line.strip().endswith(':')) and
                                   len(line.strip()) > 0]
            
            outline = '\n'.join(potential_headings[:15])  # Limit to 15 headings
            
            return {
                "status": "success",
                "message": f"Extracted outline of section {section_code}",
                "data": outline,
                "total_length": content_length,
                "has_more": True
            }
            
        else:
            # Return full content with size warning if large
            message = f"Successfully extracted section {section_code}"
            if content_length > 6000:
                message += f" (large section, {content_length} characters)"
                
            return {
                "status": "success",
                "message": message,
                "data": full_content,
                "total_length": content_length,
                "has_more": False
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": str(type(e).__name__),
            "filing_url": filing_url if 'filing_url' in locals() else None,
            "section_code": section_code if 'section_code' in locals() else None,
            "data": None
        } 