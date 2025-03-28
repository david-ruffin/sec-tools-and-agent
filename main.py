#!/usr/bin/env python3
"""
SEC Filing Analysis System
Allows users to download SEC filings as PDFs and analyze their content with natural language queries
"""

import os
import json
import argparse
from dotenv import load_dotenv
from sec_api import QueryApi, PdfGeneratorApi
from datetime import datetime
import re
from typing import Dict, List, Optional, Union
from azure.storage.blob import BlobServiceClient
from datetime import datetime

# Import Langchain components
from langchain_community.llms import OpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# Import our SEC analyzer module
from sec_analyzer import analyze_sec_filing

# Import logging utilities
from utils.logger import get_logger, log_section_boundary

# Load environment variables - works for both local .env and Azure App Settings
load_dotenv()  # This will load from .env file if it exists, but won't fail if the file is missing

# Get API keys from environment variables
SEC_API_KEY = os.getenv("SEC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not SEC_API_KEY:
    raise ValueError("SEC_API_KEY environment variable is not set")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class CompanyLookup:
    """Class for looking up company information in the SEC database."""
    
    def __init__(self, company_tickers_path: str = "reference_data/company_tickers.json", llm: Optional[BaseLanguageModel] = None):
        self.logger = get_logger()
        self.logger.info("[CompanyLookup] Initializing")
        self.company_data = self._load_company_data(company_tickers_path)
        self.llm = llm
        self.logger.info(f"[CompanyLookup] Initialized with {len(self.company_data)} companies")
        
    def _load_company_data(self, company_tickers_path: str) -> Dict:
        """Load company data from the JSON file."""
        self.logger.info(f"Loading company data from {company_tickers_path}")
        try:
            with open(company_tickers_path, 'r') as f:
                data = json.load(f)
                self.logger.info(f"Successfully loaded company data with {len(data)} entries")
                return data
        except FileNotFoundError:
            self.logger.error(f"Company tickers file not found at {company_tickers_path}")
            raise FileNotFoundError(f"Company tickers file not found at {company_tickers_path}")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in company tickers file at {company_tickers_path}")
            raise ValueError(f"Invalid JSON format in company tickers file at {company_tickers_path}")
    
    def find_company_by_name(self, company_name: str) -> Optional[Dict]:
        """Find company information by name or ticker."""
        self.logger.info(f"Finding company by name: {company_name}")
        # Use LLM to help with flexible matching
        if not self.llm:
            self.logger.debug("No LLM provided, initializing default LLM")
            self.llm = self._get_llm()
        
        # Create a prompt template for company name matching
        prompt_template = PromptTemplate(
            input_variables=["company_name"],
            template="""
            I need to match the company name "{company_name}" to one of the companies in the SEC database.
            The match should be the official company name or ticker symbol.
            Just respond with the exact company name or ticker symbol that best matches. 
            If there are multiple possible matches, pick the most likely one.
            Only respond with the exact match, nothing else.
            """
        )
        
        # Create a runnable sequence instead of LLMChain
        self.logger.debug("Creating prompt chain for company name normalization")
        chain = prompt_template | self.llm
        # Use invoke instead of run
        self.logger.debug(f"Invoking LLM to normalize company name: {company_name}")
        result = chain.invoke({"company_name": company_name})
        normalized_name = result.strip() if isinstance(result, str) else result.content.strip()
        self.logger.info(f"Normalized company name: '{company_name}' to '{normalized_name}'")
        
        # Search for the normalized name in our data
        self.logger.debug(f"Searching for normalized name '{normalized_name}' in company database")
        for _, company_info in self.company_data.items():
            if (normalized_name.lower() in company_info['title'].lower() or 
                normalized_name.lower() == company_info['ticker'].lower()):
                self.logger.info(f"Found exact match for '{normalized_name}': {company_info['title']} (CIK: {company_info['cik_str']})")
                return company_info
                
        # If no exact match, try a more flexible search
        self.logger.debug(f"No exact match found, trying flexible search with original name '{company_name}'")
        for _, company_info in self.company_data.items():
            if (company_name.lower() in company_info['title'].lower() or 
                company_name.lower() == company_info['ticker'].lower()):
                self.logger.info(f"Found flexible match for '{company_name}': {company_info['title']} (CIK: {company_info['cik_str']})")
                return company_info
        
        self.logger.warning(f"No company match found for '{company_name}'")
        return None
    
    def _get_llm(self) -> BaseLanguageModel:
        """Create a new LLM instance if one wasn't provided."""
        return ChatOpenAI(
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0
        )

class SecFilingDownloader:
    """Downloader for SEC filings with LLM enhancement and conversational interface."""
    
    def __init__(self, sec_api_key: str):
        self.logger = get_logger()
        self.logger.info("[Downloader] Initializing SecFilingDownloader")
        
        self.sec_api_key = sec_api_key
        self.query_api = QueryApi(api_key=sec_api_key)
        self.pdf_generator_api = PdfGeneratorApi(api_key=sec_api_key)
        self.logger.info("[Downloader] SEC API clients initialized")
        
        # Initialize LLM for conversation and company lookup
        self.logger.info(f"[Downloader] Initializing LLM model: {OPENAI_MODEL}")
        self.llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0
        )
        
        # Pass the LLM to CompanyLookup
        self.logger.info("[Downloader] Initializing CompanyLookup")
        self.company_lookup = CompanyLookup(llm=self.llm)
        self.logger.info("[Downloader] SecFilingDownloader initialization complete")
        
    def get_sec_filing(self, cik: str, form_type: str, year: str) -> Dict:
        """Get a filing for a given CIK, form type, and year range."""
        log_section_boundary(f"Get SEC Filing - CIK:{cik}, Form:{form_type}, Year:{year}", True)
        
        # Set up the query to find the filing based on the provided parameters
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        query = {
            "query": {
                "query_string": {
                    "query": f"cik:{cik} AND formType:\"{form_type}\" AND filedAt:[{start_date} TO {end_date}]"
                }
            },
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        self.logger.info(f"Querying SEC API for CIK:{cik}, Form:{form_type}, Year:{year}")
        self.logger.debug(f"Query: {json.dumps(query)}")
        
        # Get the filing information
        try:
            response = self.query_api.get_filings(query)
            if 'filings' in response and response['filings']:
                self.logger.info(f"Found {len(response['filings'])} filing(s)")
                for i, filing in enumerate(response['filings']):
                    self.logger.info(f"Filing {i+1}: {filing.get('formType')} filed on {filing.get('filedAt')}")
            else:
                self.logger.warning(f"No filings found for CIK:{cik}, Form:{form_type}, Year:{year}")
            
            log_section_boundary(f"Get SEC Filing - CIK:{cik}, Form:{form_type}, Year:{year}", False)
            return response
        except Exception as e:
            self.logger.error(f"Error querying SEC API: {str(e)}")
            log_section_boundary(f"Get SEC Filing - CIK:{cik}, Form:{form_type}, Year:{year}", False)
            raise
    
    def download_filing_as_pdf(self, sec_url: str, cik: str, form_type: str, year: str, filing_date: str) -> str:
        """Download filing as PDF and save to Azure Storage."""
        log_section_boundary(f"Download Filing As PDF - CIK:{cik}, Form:{form_type}", True)
        
        # Get Azure Storage connection string from environment
        storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        storage_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "filings")
        
        if not storage_connection_string:
            self.logger.error("[PDF] AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
            log_section_boundary(f"Download Filing As PDF - CIK:{cik}, Form:{form_type}", False)
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
        
        print("Converting filing to PDF format...")
        self.logger.info(f"[PDF] Generating PDF from SEC URL: {sec_url}")
        
        try:
            # Generate a PDF from the filing
            pdf_content = self.pdf_generator_api.get_pdf(sec_url)
            self.logger.info(f"[PDF] PDF generated successfully, size: {len(pdf_content)} bytes")
            
            # Create a nice filename using the company CIK, form type, and date
            date_str = filing_date.split('T')[0]
            blob_name = f"{cik}_{form_type}_{year}_{date_str}.pdf"
            self.logger.info(f"[PDF] Preparing to upload as blob: {blob_name}")
            
            # Initialize the blob service client
            blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
            container_client = blob_service_client.get_container_client(storage_container_name)
            
            # Create container if it doesn't exist
            if not container_client.exists():
                self.logger.info(f"[PDF] Creating container: {storage_container_name}")
                container_client.create_container()
            
            # Upload to Azure Storage
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(pdf_content, overwrite=True)
            
            self.logger.info(f"[PDF] PDF uploaded successfully to Azure Storage: {blob_name}")
            log_section_boundary(f"Download Filing As PDF - CIK:{cik}, Form:{form_type}", False)
            
            # Return just the blob name, as the app.py will construct the URL
            return blob_name
        except Exception as e:
            self.logger.error(f"[PDF] Error generating/uploading PDF: {str(e)}")
            log_section_boundary(f"Download Filing As PDF - CIK:{cik}, Form:{form_type}", False)
            raise
    
    def extract_parameters(self, query: str) -> Dict:
        """Extract filing parameters from a natural language query."""
        self.logger.info(f"Extracting parameters from query: '{query}'")
        
        prompt = f"""
        Extract the following parameters from this query about SEC filings:
        - Company: The company name or ticker symbol mentioned
        - Form Type: The type of SEC form (e.g., 10-K, 10-Q, 8-K, etc.)
        - Year: The year of the filing or "latest" if referring to most recent
        - Info Type: The specific information or topic the user is asking about
        
        If any parameter is missing, set its value to null.
        If the query mentions "last", "latest", "recent", or similar terms referring to the most recent filing, set Year to "latest".
        Return the parameters as a JSON object with keys 'company', 'form_type', 'year', and 'info_type'.
        
        Query: {query}
        """
        
        self.logger.debug(f"Sending parameter extraction prompt to LLM")
        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            self.logger.debug(f"LLM response for parameter extraction: {result}")
            
            if '{' in result:
                json_content = result[result.find('{'):result.rfind('}')+1]
                try:
                    json_content = json_content.replace(": None", ": null")
                    params = json.loads(json_content)
                    
                    # Handle "latest" year
                    if params.get('year') == "latest":
                        current_year = datetime.now().year
                        params['year'] = str(current_year)
                        self.logger.info(f"Converted 'latest' to current year: {params['year']}")
                    
                    self.logger.info(f"Extracted parameters: company='{params.get('company', 'None')}', form_type='{params.get('form_type', 'None')}', year='{params.get('year', 'None')}', info_type='{params.get('info_type', 'None')}'")
                    return params
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse parameters as JSON from: {json_content}")
                    print("Failed to parse parameters as JSON")
            else:
                self.logger.warning(f"No JSON found in LLM response: {result}")
            
            # Default return if parsing fails - don't return None values as the process_conversation 
            # function will handle asking for missing information
            self.logger.warning("Returning default empty parameters")
            return {"company": None, "form_type": None, "year": None, "info_type": None}
        except Exception as e:
            self.logger.error(f"Error during parameter extraction: {str(e)}")
            return {"company": None, "form_type": None, "year": None, "info_type": None}
    
    def confirm_understanding(self, params: Dict) -> str:
        """Generate a confirmation message based on extracted parameters."""
        self.logger.info("Generating confirmation message from extracted parameters")
        company = params.get('company', 'unknown company')
        form_type = params.get('form_type', 'unknown form type')
        year = params.get('year', 'unknown year')
        info_type = params.get('info_type', 'general information')
        
        # Handle cases where parameters are None
        if not company or company == 'None':
            self.logger.debug("No valid company name, using default")
            company = 'unknown company'
        if not form_type or form_type == 'None':
            self.logger.debug("No valid form type, using default")
            form_type = '10-K'  # Default to 10-K
        if not year or year == 'None':
            self.logger.debug("No valid year, using default")
            year = 'unknown year'
            
        self.logger.info(f"Confirmation parameters: company='{company}', form_type='{form_type}', year='{year}', info_type='{info_type}'")
            
        confirmation_templates = [
            f"I'll analyze {company}'s {form_type} filing from {year} to find information about {info_type}. Is that correct?",
            f"Just to confirm, you want information about {info_type} from {company}'s {form_type} in {year}?",
            f"I understand you're looking for {info_type} in {company}'s {form_type} from {year}. Is that right?"
        ]
        
        import random
        confirmation = random.choice(confirmation_templates)
        self.logger.debug(f"Selected confirmation message: '{confirmation}'")
        return confirmation
    
    def process_query_with_filing(self, query: str, company_name: str, form_type: str, year: str) -> Union[Dict, str]:
        """Process a user query about a filing after downloading the PDF."""
        log_section_boundary(f"Process Query - Query: '{query}'", True)
        
        # Step 1: Download the PDF (always do this for all requests)
        pdf_path = self.download_by_company_name(company_name, form_type, year)
        
        if not pdf_path:
            log_section_boundary(f"Process Query - Query: '{query}'", False)
            return "Could not download the filing. Please check if the filing exists."
        
        # Step 2: Get filing metadata to pass to the analyzer
        company_info = self.company_lookup.find_company_by_name(company_name)
        if not company_info:
            log_section_boundary(f"Process Query - Query: '{query}'", False)
            return f"Could not find company information for: {company_name}"
        
        cik = str(company_info['cik_str'])
        
        # Step 3: Query the SEC API to get filing data
        filing_data = self.get_sec_filing(cik, form_type, year)
        if 'filings' not in filing_data or not filing_data['filings']:
            log_section_boundary(f"Process Query - Query: '{query}'", False)
            return f"No {form_type} filings found for {company_name} (CIK: {cik}) in {year}"
        
        filing = filing_data['filings'][0]
        filing_url = filing.get('linkToFilingDetails')
        
        # Step 4: Extract parameters to get the info_type
        params = self.extract_parameters(query)
        info_type = params.get('info_type')
        
        # Step 5: Send the query and filing info to the analyzer module
        self.logger.info(f"Sending query '{query}' to analyzer with filing data")
        try:
            analysis_result = analyze_sec_filing(
                query=query,
                filing_metadata={
                    'company_name': company_name,
                    'company_title': company_info['title'],
                    'cik': cik,
                    'form_type': form_type,
                    'year': year,
                    'filing_url': filing_url,
                    'filing_date': filing.get('filedAt'),
                    'accession_no': filing.get('accessionNo'),
                    'query_topic': info_type
                },
                sec_api_key=self.sec_api_key,
                openai_api_key=OPENAI_API_KEY
            )
            
            # Step 6: Return both the analysis result and PDF path
            log_section_boundary(f"Process Query - Query: '{query}'", False)
            return {
                'analysis': analysis_result,
                'pdf_path': pdf_path
            }
        except Exception as e:
            self.logger.error(f"Error analyzing filing: {str(e)}")
            log_section_boundary(f"Process Query - Query: '{query}'", False)
            return f"Error analyzing filing: {str(e)}"
    
    def process_conversation(self) -> None:
        """Have a conversation with the user to gather SEC filing parameters and answer questions."""
        log_section_boundary("Starting Conversation Session", True)
        
        self.logger.info("[Conversation] Initializing conversation for SEC filing analysis")
        print("Welcome to the SEC Filing Analyzer. I can help you find information in SEC filings.")
        print("What would you like to know? (e.g., 'What are Apple's risk factors in their 2023 10-K?')")
        print("Type 'exit' to end the conversation.\n")
        
        while True:
            # Get user query
            user_input = input("> ")
            self.logger.info(f"[Conversation] User input: '{user_input}'")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                self.logger.info("[Conversation] User requested to exit the conversation")
                print("Goodbye!")
                break
            
            # Extract parameters from user query
            params = self.extract_parameters(user_input)
            
            # If any parameter is missing, ask specifically for it
            if not params['company'] or params['company'] == 'None':
                print("What company are you interested in?")
                params['company'] = input("> ")
                
            if not params['form_type'] or params['form_type'] == 'None':
                # Default to 10-K for this POC
                params['form_type'] = "10-K"
                print(f"Using 10-K form type for this query.")
                
            if not params['year'] or params['year'] == 'None':
                print("For which year do you need this information?")
                params['year'] = input("> ")
            
            # Confirm understanding
            confirmation = self.confirm_understanding(params)
            print(confirmation)
            confirm_input = input("(yes/no) > ").lower()
            
            if confirm_input in ['y', 'yes', 'yeah', 'correct', 'right', '']:
                # Process the query with the filing
                print(f"\nAnalyzing {params['form_type']} filing for {params['company']} from {params['year']}...")
                
                result = self.process_query_with_filing(
                    user_input,  # Original query
                    params['company'], 
                    params['form_type'], 
                    params['year']
                )
                
                if isinstance(result, dict):
                    print("\nAnalysis Result:")
                    print(result['analysis'])
                    print(f"\nYou can verify this information in the downloaded PDF: {result['pdf_path']}")
                else:
                    print(f"Error: {result}")
            else:
                print("Let's try again. What would you like to know?")
        
        log_section_boundary("Conversation Session Ended", False)
    
    def download_by_company_name(self, company_name: str, form_type: str, year: str) -> Optional[str]:
        """Download a filing using company name instead of CIK."""
        log_section_boundary(f"Download By Company Name - Company:{company_name}, Form:{form_type}, Year:{year}", True)
        
        self.logger.info(f"Looking up company: '{company_name}'")
        # Look up the company CIK
        company_info = self.company_lookup.find_company_by_name(company_name)
        
        if not company_info:
            self.logger.warning(f"Could not find company information for: '{company_name}'")
            print(f"Could not find company information for: {company_name}")
            log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
            return None
        
        cik = str(company_info['cik_str'])  # Convert CIK to string
        company_title = company_info['title']
        
        self.logger.info(f"Company found: '{company_title}' (CIK: {cik})")
        print(f"Found company: {company_title} (CIK: {cik})")
        
        try:
            self.logger.info(f"Searching for {form_type} filings for {company_title} (CIK: {cik}) in {year}")
            print(f"Searching for {form_type} filings for {company_title} (CIK: {cik}) in {year}...")
            
            # Get the filing information
            filing_data = self.get_sec_filing(cik, form_type, year)
            
            if 'filings' not in filing_data or not filing_data['filings']:
                self.logger.warning(f"No {form_type} filings found for {company_title} (CIK: {cik}) in {year}")
                print(f"No {form_type} filings found for {company_title} (CIK: {cik}) in {year}")
                log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
                return None
            
            filing = filing_data['filings'][0]
            self.logger.info(f"Selected filing: {filing['formType']} filed on {filing['filedAt']}")
            
            # Get the document URL from the filing data
            sec_url = filing.get('linkToFilingDetails')
            self.logger.info(f"Filing URL: {sec_url}")
            
            print(f"Found {form_type} filing dated {filing['filedAt']}")
            print(f"SEC URL: {sec_url}")
            
            # Download the filing as PDF
            self.logger.info("Downloading filing as PDF")
            output_path = self.download_filing_as_pdf(sec_url, cik, form_type, year, filing['filedAt'])
            
            self.logger.info(f"PDF downloaded successfully: {output_path}")
            print(f"Successfully downloaded and converted {form_type} to PDF: {output_path}")
            
            log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading filing: {str(e)}")
            print(f"Error: {str(e)}")
            log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
            return None

def main():
    # Set up logging for main function
    logger = get_logger()
    log_section_boundary("SEC Filing Analyzer Started", True)
    
    # Set up argument parser
    logger.info("[Main] Setting up argument parser")
    parser = argparse.ArgumentParser(description="Download and analyze SEC filings")
    parser.add_argument("--company", help="Company name or ticker symbol")
    parser.add_argument("--form-type", default="10-K", help="SEC Form Type (default: 10-K)")
    parser.add_argument("--year", help="Year of the filing")
    parser.add_argument("--query", help="Question about the filing to analyze")
    parser.add_argument("--interactive", action="store_true", help="Use conversational mode for filing analysis")
    
    # Parse arguments
    args = parser.parse_args()
    logger.info(f"[Main] Arguments parsed: company='{args.company}', form_type='{args.form_type}', year='{args.year}', query='{args.query}', interactive={args.interactive}")
    
    # Initialize the downloader
    logger.info("[Main] Initializing SEC Filing Analyzer")
    downloader = SecFilingDownloader(SEC_API_KEY)
    
    # Check if we should use interactive mode
    if args.interactive:
        logger.info("[Main] Starting in interactive mode")
        print("Starting conversational SEC filing analyzer...")
        downloader.process_conversation()
        logger.info(f"[Main] Interactive session completed")
        log_section_boundary("SEC Filing Analyzer Completed", False)
        return
    
    # Check if all required parameters are provided for non-interactive mode
    if not args.company or not args.year:
        logger.error("[Main] Missing required arguments for non-interactive mode")
        print("Error: --company and --year are required arguments for non-interactive mode.")
        print("Use --interactive for conversational mode.")
        log_section_boundary("SEC Filing Analyzer Completed", False)
        return
    
    # Process the query if provided
    if args.query:
        logger.info(f"[Main] Processing query: '{args.query}'")
        result = downloader.process_query_with_filing(
            args.query,
            args.company,
            args.form_type,
            args.year
        )
        
        if isinstance(result, dict):
            print("\nAnalysis Result:")
            print(result['analysis'])
            print(f"\nYou can verify this information in the downloaded PDF: {result['pdf_path']}")
        else:
            print(f"Error: {result}")
    else:
        # Just download the filing without analysis
        logger.info(f"[Main] Downloading filing without analysis")
        result = downloader.download_by_company_name(args.company, args.form_type, args.year)
        if result:
            print(f"\nFiling downloaded successfully: {result}")
        else:
            print("\nFailed to download filing.")
    
    log_section_boundary("SEC Filing Analyzer Completed", False)

if __name__ == "__main__":
    main()