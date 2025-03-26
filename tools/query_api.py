from sec_api import QueryApi
import os
from dotenv import load_dotenv
import inspect
from tools.logger import log_api_call

# Load API key
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

@log_api_call
def sec_filing_search(query):
    """
    Search for SEC filings using Elasticsearch query syntax.
    
    Args:
        query: String query or dict with query parameters
              Common fields: ticker, cik, companyName, formType, filedAt
              
    Examples:
        - "ticker:AAPL AND formType:\"10-K\""
        - {"query": "ticker:MSFT", "from": "0", "size": "10"}
              
    Returns:
        Dict with filing results or error information
    """
    try:
        # Initialize the API client
        query_api = QueryApi(api_key=SEC_API_KEY)
        
        # Format the query if it's not already a dictionary
        if isinstance(query, str):
            query = {
                "query": query,
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }
        elif isinstance(query, dict):
            # Fix for nested 'query' parameter
            if "query" in query and isinstance(query["query"], dict):
                # Convert nested query dictionary to Elasticsearch format
                nested_query = query["query"]
                query_parts = []
                
                # Handle common fields
                if "ticker" in nested_query:
                    query_parts.append(f"ticker:{nested_query['ticker']}")
                
                if "cik" in nested_query:
                    query_parts.append(f"cik:{nested_query['cik']}")
                    
                if "company_name" in nested_query:
                    query_parts.append(f"companyName:\"{nested_query['company_name']}\"")
                
                if "form_type" in nested_query:
                    query_parts.append(f"formType:\"{nested_query['form_type']}\"")
                
                # Date handling
                if "filing_year" in nested_query:
                    year = nested_query["filing_year"]
                    query_parts.append(f"filedAt:[{year}-01-01 TO {year}-12-31]")
                
                # Build full query string
                if query_parts:
                    query["query"] = " AND ".join(query_parts)
                else:
                    return {
                        "status": "error",
                        "message": "Could not convert nested query to Elasticsearch format",
                        "data": None
                    }
            elif "query" not in query:
                # Try to build a query from common fields
                query_parts = []
                
                # Handle direct fields at the top level
                if "ticker" in query:
                    query_parts.append(f"ticker:{query['ticker']}")
                
                if "cik" in query:
                    query_parts.append(f"cik:{query['cik']}")
                    
                if "company_name" in query:
                    query_parts.append(f"companyName:\"{query['company_name']}\"")
                
                if "form_type" in query:
                    query_parts.append(f"formType:\"{query['form_type']}\"")
                
                # Date handling
                if "filing_year" in query:
                    year = query["filing_year"]
                    query_parts.append(f"filedAt:[{year}-01-01 TO {year}-12-31]")
                
                # Build full query 
                if query_parts:
                    query = {
                        "query": " AND ".join(query_parts),
                        "from": "0",
                        "size": "10",
                        "sort": [{"filedAt": {"order": "desc"}}]
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Query dictionary must contain a 'query' field or recognizable parameters",
                        "data": None
                    }
        
        # Call the API
        result = query_api.get_filings(query)
        
        # Verify we got results
        if not result or "filings" not in result or not result["filings"]:
            return {
                "status": "no_results",
                "message": "No SEC filings found matching the criteria",
                "query": query,
                "data": None
            }
        
        # Return standardized response
        return {
            "status": "success",
            "message": f"Found {len(result['filings'])} filings",
            "query": query,
            "data": result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": str(type(e).__name__),
            "data": None
        }