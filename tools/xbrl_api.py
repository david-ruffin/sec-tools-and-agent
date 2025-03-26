from sec_api import XbrlApi
import os
from dotenv import load_dotenv
from tools.logger import log_api_call

# Load API key
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

@log_api_call
def sec_xbrl_extractor(htm_url=None, accession_number=None, return_format="json", statement_type=None, metrics=None):
    """
    Extract structured financial data from SEC filings in XBRL format.
    
    Args:
        htm_url: URL to the SEC filing HTML document
        accession_number: Alternative to htm_url, SEC filing accession number
        return_format: Output format, either "json" (default) or "raw"
        statement_type: Specific statement to retrieve ("IncomeStatements", "BalanceSheets", "CashFlowStatements", etc.)
        metrics: List of specific metrics to retrieve within a statement (e.g., ["Revenue", "NetIncome"])
    
    Returns:
        Dict with XBRL financial data or error information
    """
    try:
        if not htm_url and not accession_number:
            return {
                "status": "error", 
                "message": "Must provide either htm_url or accession_number",
                "error_type": "ValueError",
                "data": None
            }
        
        xbrl_api = XbrlApi(api_key=SEC_API_KEY)
        
        # Process based on input parameters
        result = None
        source_info = {}
        
        # Get the raw XBRL data if requested
        if return_format == "raw":
            if htm_url:
                source_info["filing_url"] = htm_url
                raw_data = xbrl_api.get_raw_xbrl(htm_url=htm_url)
                result = {"xbrl_raw": raw_data}
            else:
                source_info["accession_number"] = accession_number
                raw_data = xbrl_api.get_raw_xbrl(accession_no=accession_number)
                result = {"xbrl_raw": raw_data}
                
            # Return early - no filtering for raw format
            return {
                "status": "success",
                "message": "Successfully extracted raw XBRL data",
                **source_info,
                "data": result
            }
        
        # Get the JSON data
        if htm_url:
            source_info["filing_url"] = htm_url
            full_data = xbrl_api.xbrl_to_json(htm_url=htm_url)
        else:
            source_info["accession_number"] = accession_number
            full_data = xbrl_api.xbrl_to_json(accession_no=accession_number)
        
        # If no statement type is specified, return just the available statements
        if not statement_type:
            # Just return the keys of the full data (statement types)
            available_statements = list(full_data.keys())
            return {
                "status": "success",
                "message": "Available financial statements. Use statement_type parameter to select one.",
                **source_info,
                "available_statements": available_statements,
                "data": None
            }
        
        # Map common statement name variations as mentioned in the docs
        # "Variants such as ConsolidatedStatementsofOperations or ConsolidatedStatementsOfLossIncome 
        # are automatically standardized to their root name, e.g. StatementsOfIncome"
        statement_name_map = {
            # Income statement variations
            "incomestatements": "StatementsOfIncome",
            "statementsofoperations": "StatementsOfIncome",
            "statementsofincome": "StatementsOfIncome",
            "consolidatedstatementsofoperations": "StatementsOfIncome",
            "consolidatedstatementsofincome": "StatementsOfIncome",
            "consolidatedstatementsoflossincome": "StatementsOfIncome",
            "operationsstatements": "StatementsOfIncome",
            
            # Balance sheet variations
            "balancesheets": "BalanceSheets",
            "consolidatedbalancesheets": "BalanceSheets",
            "statementsoffinancialposition": "BalanceSheets",
            "consolidatedstatementsoffinancialposition": "BalanceSheets",
            
            # Cash flow variations
            "cashflowstatements": "StatementsOfCashFlows",
            "statementofcashflows": "StatementsOfCashFlows",
            "statementsofcashflows": "StatementsOfCashFlows",
            "consolidatedstatementsofcashflows": "StatementsOfCashFlows",
            
            # Comprehensive income variations
            "comprehensiveincome": "StatementsOfComprehensiveIncome",
            "statementsofcomprehensiveincome": "StatementsOfComprehensiveIncome",
            "consolidatedstatementsofcomprehensiveincome": "StatementsOfComprehensiveIncome"
        }
        
        # Try to map the provided statement_type to a standardized name
        normalized_statement_type = statement_name_map.get(statement_type.lower().replace(" ", ""), statement_type)
        
        # If the normalized statement type is not found, try to find a close match
        if normalized_statement_type not in full_data:
            # Look for partial matches in the available statement names
            possible_matches = []
            for key in full_data.keys():
                # Check if the normalized statement type is a substring of any available key
                if (normalized_statement_type.lower() in key.lower() or 
                    any(part.lower() in key.lower() for part in normalized_statement_type.split("Statements"))):
                    possible_matches.append(key)
            
            # If we found possible matches, use the first one
            if possible_matches:
                normalized_statement_type = possible_matches[0]
        
        # If statement type is still not found, return available statements
        if normalized_statement_type not in full_data:
            return {
                "status": "no_results",
                "message": f"Statement type '{statement_type}' not found in the filing",
                "available_statements": list(full_data.keys()),
                **source_info,
                "data": None
            }
        
        # Get the requested statement
        statement_data = full_data[normalized_statement_type]
        
        # If metrics are specified, filter to just those metrics
        if metrics and isinstance(metrics, list) and len(metrics) > 0:
            # Convert metrics to lowercase for case-insensitive matching
            metrics_lower = [m.lower() for m in metrics]
            
            # Find matching keys in a case-insensitive manner
            filtered_data = {}
            for key in statement_data:
                # Check if any of the requested metrics is in the key name
                if any(metric.lower() in key.lower() for metric in metrics):
                    filtered_data[key] = statement_data[key]
            
            # If no matching metrics were found
            if not filtered_data:
                return {
                    "status": "no_results",
                    "message": f"No matching metrics found for {metrics} in {normalized_statement_type}",
                    "available_metrics": list(statement_data.keys()),
                    **source_info,
                    "data": None
                }
            
            result = filtered_data
        else:
            # Return the entire statement
            result = statement_data
        
        # Return the filtered data
        return {
            "status": "success",
            "message": f"Successfully extracted {normalized_statement_type}" + 
                       (f" filtered by {metrics}" if metrics else ""),
            **source_info,
            "data": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": str(type(e).__name__),
            "data": None,
            **({"filing_url": htm_url} if htm_url else {"accession_number": accession_number})
        }
