#!/usr/bin/env python3
"""
SEC Agent - Demonstrating multiple tool usage with SEC-API

This script shows how an agent can use multiple SEC-API tools together by:
1. Finding recent 10-K filings for a specified company using the Query API
2. Extracting the Risk Factors section from those filings using the Extractor API

Usage: python sec_agent.py
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sec_api import QueryApi, ExtractorApi

# Load environment variables from .env file
load_dotenv()

class SecAgent:
    def __init__(self):
        # Get API key from environment variables
        self.api_key = os.getenv("SEC_API_KEY")
        if not self.api_key:
            raise ValueError("SEC_API_KEY environment variable not found")
        
        # Initialize both tools
        self.query_api = QueryApi(self.api_key)
        self.extractor_api = ExtractorApi(self.api_key)
    
    def search_recent_filings(self, ticker, form_type, num_years=3):
        """
        Search for recent filings of a specific type for a company
        """
        # Calculate date range (last N years)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365*num_years)).strftime("%Y-%m-%d")
        date_range = f"{start_date} TO {end_date}"
        
        print(f"Searching for {form_type} filings for {ticker} from {start_date} to {end_date}...")
        
        # Construct and execute search query
        query = {
            "query": f"ticker:{ticker} AND formType:\"{form_type}\" AND filedAt:[{date_range}]",
            "from": "0",
            "size": "5",  # Limiting to 5 most recent filings
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        return self.query_api.get_filings(query)
    
    def extract_section(self, filing_url, section_code):
        """
        Extract a specific section from a filing
        """
        print(f"Extracting section {section_code} from {filing_url}...")
        return self.extractor_api.get_section(filing_url, section_code, "text")
    
    def search_and_extract(self, ticker, form_type, section_code, num_years=3):
        """
        Combine both tools: search for filings and extract sections
        """
        # Step 1: Use the Query API to search for filings
        filings_data = self.search_recent_filings(ticker, form_type, num_years)
        
        if not filings_data or "filings" not in filings_data or not filings_data["filings"]:
            print(f"No {form_type} filings found for {ticker} in the specified time range.")
            return []
        
        # Step 2: Use the Extractor API to get sections from each filing
        results = []
        for filing in filings_data["filings"]:
            filing_url = filing.get("linkToFilingDetails")
            filing_date = filing.get("filedAt", "Unknown date")
            
            if filing_url:
                try:
                    section_text = self.extract_section(filing_url, section_code)
                    
                    # Add result if section was found
                    if section_text and len(section_text) > 50:  # Basic check that we got content
                        results.append({
                            "ticker": ticker,
                            "form_type": form_type,
                            "filing_date": filing_date,
                            "filing_url": filing_url,
                            "section_code": section_code,
                            "section_text": section_text[:500] + "..." if len(section_text) > 500 else section_text
                        })
                        print(f"Successfully extracted section {section_code} from {filing_date} filing")
                    else:
                        print(f"Section {section_code} not found or empty in {filing_date} filing")
                except Exception as e:
                    print(f"Error extracting section from {filing_date} filing: {str(e)}")
        
        return results

def main():
    """Main function demonstrating the SEC Agent usage"""
    try:
        # Initialize the agent
        agent = SecAgent()
        
        # Example: Find AAPL's recent 10-K filings and extract Risk Factors (section 1A)
        ticker = "AAPL"
        results = agent.search_and_extract(
            ticker=ticker,
            form_type="10-K",
            section_code="1A",
            num_years=2
        )
        
        # Display results
        print(f"\nFound {len(results)} filings with Risk Factors sections for {ticker}:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Filing Date: {result['filing_date']}")
            print(f"URL: {result['filing_url']}")
            print(f"Preview of section {result['section_code']} (Risk Factors):")
            print(f"{result['section_text'][:300]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 