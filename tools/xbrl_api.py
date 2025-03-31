from sec_api import XbrlApi
import os
from dotenv import load_dotenv
from tools.logger import log_api_call

# Load API key
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

@log_api_call
def sec_xbrl_extractor(filing_url, statement_type=None, metrics=None, original_query=None):
    """
    Extract XBRL financial data from an SEC filing.
    
    Args:
        filing_url: URL to the SEC filing
        statement_type: Optional specific statement to extract (e.g., 'StatementsOfIncome')
        metrics: Optional list of specific metrics to extract
        original_query: The original user query
    
    Returns:
        Dict with extracted XBRL data and status information
    """
    try:
        # Validate inputs
        if not filing_url:
            return {
                "status": "error",
                "message": "Filing URL is required",
                "data": None
            }
        
        # Initialize the XBRL API
        xbrl_converter = XbrlApi(api_key=SEC_API_KEY)
        
        # Extract XBRL data
        xbrl_json = xbrl_converter.xbrl_to_json(htm_url=filing_url)
        
        # Check if extraction was successful
        if not xbrl_json or "error" in xbrl_json:
            return {
                "status": "error",
                "message": f"XBRL extraction failed: {xbrl_json.get('error', 'Unknown error')}",
                "data": None
            }
        
        # Handle no statement_type specified - return available statements
        if not statement_type:
            # Get available statements
            available_statements = {}
            
            # Parse results to get statement names at the top level
            for key in xbrl_json.keys():
                if key not in ["CIK", "EntityRegistrantName", "CurrentFiscalYearEndDate", "fact"]:
                    available_statements[key] = xbrl_json[key].keys() if isinstance(xbrl_json[key], dict) else "Available"
            
            # Create metadata
            metadata = {
                "company": xbrl_json.get("EntityRegistrantName", ""),
                "cik": xbrl_json.get("CIK", ""),
                "fiscal_year_end": xbrl_json.get("CurrentFiscalYearEndDate", "")
            }
            
            # Prepare text content for RAG processing
            raw_text = f"Company: {metadata['company']}\nCIK: {metadata['cik']}\nFiscal Year End: {metadata['fiscal_year_end']}\n\n"
            raw_text += "Available Financial Statements:\n"
            for stmt, details in available_statements.items():
                raw_text += f"- {stmt}: {list(details) if isinstance(details, dict) else details}\n"
            
            return {
                "status": "success",
                "message": "XBRL extracted successfully. No statement_type specified, returning available statements.",
                "data": {
                    "available_statements": available_statements,
                    "metadata": metadata
                },
                "raw_text": raw_text
            }
        
        # Check if the requested statement exists
        if statement_type not in xbrl_json:
            return {
                "status": "error",
                "message": f"Statement type '{statement_type}' not found in XBRL data. Available types: {', '.join(xbrl_json.keys())}",
                "data": None
            }
        
        # Extract the specific statement requested
        statement_data = xbrl_json[statement_type]
        
        # Extract only requested metrics if specified
        if metrics and isinstance(metrics, list):
            filtered_data = {}
            for period, values in statement_data.items():
                filtered_data[period] = {}
                for metric in metrics:
                    if metric in values:
                        filtered_data[period][metric] = values[metric]
            statement_data = filtered_data
        
        # Create metadata
        metadata = {
            "company": xbrl_json.get("EntityRegistrantName", ""),
            "cik": xbrl_json.get("CIK", ""),
            "fiscal_year_end": xbrl_json.get("CurrentFiscalYearEndDate", ""),
            "statement_type": statement_type,
            "metrics_requested": metrics
        }
        
        # Prepare text content for RAG processing
        raw_text = f"Company: {metadata['company']}\nCIK: {metadata['cik']}\nFiscal Year End: {metadata['fiscal_year_end']}\n\n"
        raw_text += f"Statement Type: {statement_type}\n\n"
        
        # Add the data in a format conducive to text processing
        for period, values in statement_data.items():
            raw_text += f"Period: {period}\n"
            for metric, value in values.items():
                if isinstance(value, dict) and "value" in value:
                    raw_text += f"  {metric}: {value['value']}\n"
                else:
                    raw_text += f"  {metric}: {value}\n"
            raw_text += "\n"
        
        return {
            "status": "success",
            "message": f"XBRL data extracted for statement '{statement_type}'",
            "data": statement_data,
            "metadata": metadata,
            "raw_text": raw_text
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": str(type(e).__name__),
            "data": None
        }
