from sec_api import QueryApi
import os
from dotenv import load_dotenv
import inspect
from tools.logger import log_api_call

# Load API key
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

@log_api_call
def sec_filing_search(query, original_query=None):
    """
    Search for SEC filings using the Query API.
    
    Args:
        query: The Elasticsearch query string (e.g., 'ticker:AAPL AND formType:"10-K"')
               or dictionary with full query parameters
        original_query: The original user query
    
    Returns:
        Dict with search results and status information
    """
    try:
        # Initialize the Query API
        query_api = QueryApi(api_key=SEC_API_KEY)
        
        # Process based on input type
        if isinstance(query, str):
            # If given a string, create a default query structure with just this query
            query_obj = {
                "query": query,
                "from": "0",
                "size": "10",
                "sort": [{"filedAt": {"order": "desc"}}]
            }
        elif isinstance(query, dict) and "query" in query:
            # User provided a complete query object
            query_obj = query
        else:
            return {
                "status": "error",
                "message": "Query must be a string or a dictionary with a 'query' key",
                "data": None
            }
            
        # Execute the query
        response = query_api.get_filings(query_obj)
        
        # Return results with proper format
        if not response or "filings" not in response:
            return {
                "status": "error",
                "message": "No response from SEC API",
                "data": None
            }
            
        filings = response.get("filings", [])
        
        # Check for empty results
        if not filings:
            return {
                "status": "no_results",
                "message": f"No filings found for query: {query_obj['query']}",
                "query": query_obj,
                "data": None
            }
            
        # Prepare text content for RAG processing
        raw_text = f"SEC filing search results for query: {query_obj['query']}\n\n"
        
        for idx, filing in enumerate(filings):
            raw_text += f"Result {idx+1}:\n"
            raw_text += f"Company: {filing.get('companyName', 'N/A')}\n"
            raw_text += f"Ticker: {filing.get('ticker', 'N/A')}\n"
            raw_text += f"Form Type: {filing.get('formType', 'N/A')}\n"
            raw_text += f"Filed At: {filing.get('filedAt', 'N/A')}\n"
            raw_text += f"Period of Report: {filing.get('periodOfReport', 'N/A')}\n"
            raw_text += f"Description: {filing.get('description', 'N/A')}\n"
            raw_text += f"Link: {filing.get('linkToFilingDetails', 'N/A')}\n\n"
        
        # Return successful result
        return {
            "status": "success",
            "message": f"Found {len(filings)} filings",
            "query": query_obj,
            "data": response,
            "raw_text": raw_text
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": str(type(e).__name__),
            "data": None
        }