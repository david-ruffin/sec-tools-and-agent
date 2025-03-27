import os
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Type, Union
import logging # <-- Add logging
import datetime # <-- Add datetime
import pytz # <-- Add pytz
import sys # <-- Import sys to handle command-line arguments
from enum import Enum

# Use Pydantic v2 directly
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Import LangChain Callback Components ---
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnablePassthrough # Add this import

# --- Import SEC API Clients ---
try:
    from sec_api import (
        QueryApi,
        ExtractorApi,
        MappingApi,
        XbrlApi,
    )
except ImportError:
    print("Please install the sec-api library: pip install sec-api")
    exit()

# --- Load API Keys Securely ---
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Setup Logging ---

def setup_logging():
    """Configures logging to file and console with PST timestamps."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Define PST timezone
    pst_tz = pytz.timezone('America/Los_Angeles')

    # Generate timestamped filename in PST
    pst_now = datetime.datetime.now(pst_tz)
    log_filename = pst_now.strftime(f"{log_dir}/log_%Y-%m-%d_%H-%M-%S_%Z.log")

    # Create formatter with PST timezone awareness
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(log_format)
    # Set converter for log record timestamps to use PST
    logging.Formatter.converter = lambda *args: datetime.datetime.now(pst_tz).timetuple()

    # Create file handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get the root logger and configure it
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set desired log level (e.g., INFO, DEBUG)
    # Avoid adding handlers multiple times if setup_logging is called again
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler) # Keep console output as well

    # Silence overly verbose libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Call setup_logging() early
setup_logging()
logger = logging.getLogger(__name__) # Get a logger for this module

logger.info("Logging initialized.") # Log initialization

if not SEC_API_KEY:
    logger.error("SEC_API_KEY environment variable not set.")
    raise ValueError("SEC_API_KEY environment variable not set.")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set.")
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

logger.info("API keys loaded.")

# --- Initialize SEC API Clients ---
query_api = QueryApi(api_key=SEC_API_KEY)
extractor_api = ExtractorApi(api_key=SEC_API_KEY)
mapping_api = MappingApi(api_key=SEC_API_KEY)
xbrl_api = XbrlApi(api_key=SEC_API_KEY)
logger.info("SEC API clients initialized.")

# --- Re-add Enum Definition ---
class SortOrderEnum(str, Enum):
    asc = "asc"
    desc = "desc"

# --- Re-add SortCriterion Definition ---
class SortCriterion(BaseModel):
    field_name: str = Field(description="The name of the field to sort by (e.g., 'filedAt', 'periodOfReport').")
    order: SortOrderEnum = Field(description="The sort order: 'asc' for ascending, 'desc' for descending.")

# --- Define Tool Input Schemas using Pydantic v2 ---

class SearchFilingsInput(BaseModel):
    query: str = Field(description="Mandatory. The search query string using SEC-API query syntax (similar to Lucene). Examples: 'ticker:AAPL AND formType:\"10-K\"', 'formType:\"8-K\" AND items:\"1.01\" AND filedAt:[2023-01-01 TO 2023-12-31]'. Use specific fields like ticker, cik, formType, filedAt, periodOfReport, items, sic, etc.")
    start: int = Field(0, description="Optional. The starting index for results (pagination). Default is 0.")
    size: int = Field(10, description="Optional. The number of results to return (pagination). Default is 10. Max is typically 200, check sec-api docs.")
    # --- Reverted Sort Structure to use SortCriterion ---
    sort: Optional[List[SortCriterion]] = Field(None, description="Optional. How to sort results. Provide a list of sort criteria objects, each specifying a 'field_name' and 'order' ('asc' or 'desc'). Example: [{'field_name': 'filedAt', 'order': 'desc'}]")

class ExtractSectionInput(BaseModel):
    filing_url: str = Field(description="Mandatory. The URL of the filing's HTML page on sec.gov. Example: 'https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-10k_20201231.htm'")
    section: str = Field(description="Mandatory. The section to extract. Use the specific identifiers listed in the prompt based on the form type (e.g., '1A', '7', '8' for 10-K; 'part1item1', 'part2item1a' for 10-Q; '1-1', '5-2' for 8-K).")
    return_type: str = Field("text", description="Optional. The format of the returned section ('text' or 'html'). Default is 'text'.")
    user_query: Optional[str] = Field(None, description="Optional. The original user query that prompted this extraction. Used to retrieve relevant chunks if the section is very long.")

class MapIdentifiersInput(BaseModel):
    identifier_type: str = Field(description="Mandatory. The type of identifier to resolve. Must be one of: 'ticker', 'cik', 'cusip', 'name', 'exchange'.")
    value: str = Field(description="Mandatory. The value of the identifier to resolve. Example: 'AAPL' for ticker, '320193' for CIK.")

class GetXbrlDataInput(BaseModel):
    htm_url: Optional[str] = Field(None, description="Optional. URL of the filing's primary HTML document (.htm). Use this OR xbrl_url OR accession_no.")
    xbrl_url: Optional[str] = Field(None, description="Optional. URL of the filing's XBRL instance document (.xml). Use this OR htm_url OR accession_no.")
    accession_no: Optional[str] = Field(None, description="Optional. Accession number of the filing (e.g., '0001564590-21-004599'). Use this OR htm_url OR xbrl_url.")

# --- Define Custom Tools ---

class SearchFilingsTool(BaseTool):
    name: str = "search_sec_filings"
    # --- Update description slightly to match SortCriterion example format ---
    # --- AND add explicit instruction on which URL to use for extraction ---
    description: str = """
    Searches and filters SEC EDGAR filings based on various criteria like ticker, CIK, company name,
    form type (e.g., "10-K", "10-Q", "8-K", "13F-HR"), filing date, reporting period, SIC code, 8-K items, etc.
    Uses the SEC-API query syntax (like Lucene). Returns metadata about matching filings, including links.
    Crucial for finding relevant filings before extracting data.
    Use `filedAt:[YYYY-MM-DD TO YYYY-MM-DD]` for specific date ranges.
    Use `periodOfReport:[YYYY-MM-DD TO YYYY-MM-DD]` for report period ranges.
    Combine criteria using AND/OR. Example: 'ticker:MSFT AND formType:"10-K" AND filedAt:[2023-01-01 TO 2023-12-31]'
    **IMPORTANT: When the user asks for the 'latest', 'most recent', or similar terms for a filing, you MUST include `sort: [{{'field_name': 'filedAt', 'order': 'desc'}}]` in your parameters and usually set `size: 1` to get only the single most recent result.** The tool defaults to sorting by 'filedAt' descending if no sort is specified, but explicitly setting it for 'latest' queries ensures correctness.
    **OUTPUT USAGE: After finding a filing, use the 'linkToFilingDetails' URL or the primary document URL (often ending in .htm) found within 'documentFormatFiles' as the 'filing_url' input for the 'extract_filing_section' tool.**
    """
    args_schema: Type[BaseModel] = SearchFilingsInput

    # --- Adjust _run to convert SortCriterion list to API format ---
    def _run(self, query: str, start: int = 0, size: int = 10, sort: Optional[List[SortCriterion]] = None) -> str:
        """Use the tool."""
        try:
            search_query = {
                "query": query,
                "from": str(start),
                "size": str(size),
            }
            if sort:
                # Convert the validated List[SortCriterion] to the format sec-api expects
                api_sort = [{criterion.field_name: {"order": criterion.order.value}} for criterion in sort]
                search_query["sort"] = api_sort
                # Log the original SortCriterion objects for clarity
                try:
                    sort_repr = json.dumps([criterion.dict() for criterion in sort])
                except Exception:
                    sort_repr = "[unserializable sort object]"
                logger.info(f"SearchFilingsTool called with query: {query}, start: {start}, size: {size}, sort: {sort_repr}")
            else:
                search_query["sort"] = [{"filedAt": {"order": "desc"}}] # Default sort
                logger.info(f"SearchFilingsTool called with query: {query}, start: {start}, size: {size}, sort: Default (filedAt desc)")

            logger.debug(f"Constructed API search query: {search_query}")
            filings = query_api.get_filings(search_query)
            logger.debug(f"API response received (type: {type(filings).__name__})")

            # --- ADDED TYPE CHECKING ---
            if not isinstance(filings, dict):
                error_message = f"Error: API call returned unexpected type '{type(filings).__name__}' instead of dict. Value: {filings}. This might indicate an invalid query or unsupported sort field."
                # print(f"\n>> {error_message}") // Replaced
                logger.error(error_message)
                return error_message
            # --- END TYPE CHECKING ---

            total_found = filings.get('total', {}).get('value', 0)
            # print(f">> Found {total_found} filings.") // Replaced
            logger.info(f"API reported {total_found} total matching filings.")

            # --- ADDED CHECK FOR API-LEVEL ERRORS ---
            if 'error' in filings:
                 error_detail = filings.get('error', 'Unknown API error')
                 # print(f"\n>> API returned an error: {error_detail}") // Replaced
                 logger.error(f"API returned an error: {error_detail}")
                 return f"Error from sec-api: {error_detail}"
            # --- END CHECK FOR API-LEVEL ERRORS ---

            results = filings.get('filings', [])
            logger.info(f"Returning {len(results)} filings (limited by size parameter).")
            # Return limited results
            return json.dumps(results[:size]) # Ensure we respect size param here too

        except Exception as e:
            # Catch other potential exceptions during the process
            # print(f"\n>> Exception in SearchFilingsTool: {e}") // Replaced
            try:
                sort_repr = repr(sort) # Keep simple repr for exception logging
            except Exception:
                sort_repr = "[unrepresentable sort object]"
            error_msg = f"Exception in SearchFilingsTool: {e}. Query: {query}, Sort: {sort_repr}. Check query syntax and parameters."
            logger.exception(error_msg) # Use logger.exception to include stack trace
            return f"Error executing search_sec_filings: {e}. Query: {query}, Sort: {sort_repr}. Check query syntax and parameters."

class ExtractSectionTool(BaseTool):
    name: str = "extract_filing_section"
    description: str = """
    Extracts specific sections from a given SEC filing URL based on the form type (10-K, 10-Q, 8-K).
    Requires the direct URL to the filing's HTML document (prefer .htm over ix?doc= URLs) and the correct section identifier (e.g., '1A', 'part1item1', '5-2').
    Use this *after* finding the relevant filing URL with 'search_sec_filings'.
    If the section is very long, it attempts to retrieve chunks relevant to the user's original query (pass this as 'user_query' parameter).
    Returns the cleaned text content of the specified section or relevant excerpts.
    """
    args_schema: Type[BaseModel] = ExtractSectionInput
    # Add a field to potentially hold the overall input query
    input_query: Optional[str] = None

    def _run(self, filing_url: str, section: str, return_type: str = "text", user_query: Optional[str] = None) -> str:
        """Use the tool. Implements basic RAG for long sections."""
        # --- URL Preference ---
        if "ix?doc=" in filing_url:
            logger.warning(f"Received iXBRL URL ({filing_url}). Extraction might be less reliable or produce very large output. Prefer standard .htm URLs.")
        # --- End URL Preference ---

        if return_type != "text":
            logger.warning("RAG chunking only implemented for return_type='text'. Falling back to basic extraction.")
            # Fallback to original behavior for non-text types
            try:
                logger.info(f"ExtractSectionTool called (non-text) for URL: {filing_url}, Section: {section}, ReturnType: {return_type}")
                content = extractor_api.get_section(filing_url, section, return_type)
                logger.info(f"Extracted section content (length: {len(content)}).")
                # No truncation here for HTML, let the LLM handle it if needed, or consider adding later
                return content
            except Exception as e:
                error_msg = f"Exception in ExtractSectionTool (non-text): {e}. URL: {filing_url}, Section: {section}."
                logger.exception(error_msg)
                return f"Error executing extract_filing_section: {e}. Ensure URL is correct and section identifier is valid for the form type."

        # --- Text Extraction with RAG logic ---
        try:
            logger.info(f"ExtractSectionTool called for URL: {filing_url}, Section: {section}, ReturnType: {return_type}")
            content = extractor_api.get_section(filing_url, section, "text") # Always get text for RAG
            content_len = len(content)
            logger.info(f"Extracted section content (length: {content_len}).")

            limit = 15000 # Define the threshold for triggering RAG and the final output limit

            if content_len <= limit:
                logger.info("Content length within limit, returning full text.")
                return content
            else:
                logger.warning(f"Content length {content_len} exceeds limit {limit}. Applying RAG chunk retrieval.")

                # 1. Chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, # Adjust chunk size as needed
                    chunk_overlap=100, # Adjust overlap as needed
                    length_function=len,
                )
                chunks = text_splitter.split_text(content)
                logger.info(f"Split content into {len(chunks)} chunks.")

                if not chunks:
                    logger.warning("Text splitting resulted in no chunks. Returning truncated original content.")
                    return content[:limit] + "... (truncated)"

                # 2. Embedding & Vector Store
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                    logger.info("Initializing FAISS vector store...")
                    vector_store = FAISS.from_texts(chunks, embeddings)
                    logger.info("FAISS vector store created successfully.")
                except Exception as e:
                    logger.exception(f"Failed to create vector store: {e}. Returning truncated original content.")
                    return content[:limit] + "... (truncated)"

                # 3. Retrieval
                retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Retrieve top 10 chunks
                # --- MODIFIED: Use user_query if provided, otherwise try the overall input_query ---
                retrieval_query = user_query
                if not retrieval_query:
                    logger.warning("No specific 'user_query' parameter provided for RAG retrieval.")
                    if self.input_query:
                        logger.info("Using overall agent input query as fallback for RAG.")
                        retrieval_query = self.input_query
                    else:
                        # Basic fallback: use section name or generic term
                        logger.warning("No overall agent input query available. Using section name as fallback query.")
                        section_map = { "1A": "risk factors", "7": "management discussion analysis", "8": "financial statements footnotes", "part1item1": "financial statements footnotes", "part1item2": "management discussion analysis", "part2item1a": "risk factors", "1C": "cybersecurity"}
                        retrieval_query = section_map.get(section, section) # Use section ID if not in map
                # --- END MODIFICATION ---

                logger.info(f"Performing retrieval with query: '{retrieval_query}'")
                try:
                    retrieved_docs = retriever.invoke(retrieval_query)
                except Exception as e:
                    logger.exception(f"Failed during retrieval: {e}. Returning truncated original content.")
                    return content[:limit] + "... (truncated)"

                # 4. Combine and Return Relevant Chunks
                combined_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                combined_len = len(combined_text)
                logger.info(f"Retrieved {len(retrieved_docs)} chunks, combined length: {combined_len}.")

                if combined_len > limit:
                    logger.warning(f"Combined retrieved text length {combined_len} exceeds limit {limit}. Truncating retrieved text.")
                    final_text = combined_text[:limit] + "... (retrieved excerpts truncated)"
                else:
                    final_text = combined_text + "\n\n...(Retrieved relevant excerpts)"

                return final_text

        except Exception as e:
            error_msg = f"Exception in ExtractSectionTool: {e}. URL: {filing_url}, Section: {section}. Ensure URL is correct and section identifier is valid."
            logger.exception(error_msg)
            # Fallback to original error message format
            return f"Error executing extract_filing_section: {e}. Ensure URL is correct and section identifier is valid for the form type."

class MapIdentifiersTool(BaseTool):
    name: str = "map_sec_identifiers"
    description: str = """
    Resolves or maps between different SEC identifiers like Ticker, CIK, CUSIP, Company Name, or Exchange.
    Useful for finding the CIK for a given Ticker or vice-versa before searching filings.
    Specify the type of identifier you have ('ticker', 'cik', 'cusip', 'name', 'exchange') and its value.
    Returns details of the matching entity/entities.
    """
    args_schema: Type[BaseModel] = MapIdentifiersInput

    def _run(self, identifier_type: str, value: str) -> str:
        """Use the tool."""
        try:
            # print(f"\n>> Calling MapIdentifiersTool for Type: {identifier_type}, Value: {value}") // Replaced
            logger.info(f"MapIdentifiersTool called for Type: {identifier_type}, Value: {value}")
            result = mapping_api.resolve(identifier_type, value)
            # print(f">> Mapping result: {result}") // Replaced
            logger.info(f"Mapping result received.")
            logger.debug(f"Mapping result data: {result}")
            return json.dumps(result)
        except Exception as e:
            # print(f"\n>> Error in MapIdentifiersTool: {e}") // Replaced
            error_msg = f"Exception in MapIdentifiersTool: {e}. Type: {identifier_type}, Value: {value}. Check identifier type and value."
            logger.exception(error_msg)
            return f"Error executing map_sec_identifiers: {e}. Check identifier type and value."

class GetXbrlDataTool(BaseTool):
    name: str = "get_xbrl_data_as_json"
    description: str = """
    Parses and converts XBRL data from a 10-K, 10-Q, or other XBRL-enabled filing into a structured JSON format.
    Provide *one* of: the filing's HTML URL (`htm_url`), the XBRL instance URL (`xbrl_url`), or the filing's `accession_no`.
    Returns standardized financial statements (Income Statement, Balance Sheet, Cash Flow) and other tagged data found in the XBRL.
    Useful for extracting specific financial figures (like Revenue, Inventory, Options Granted if tagged).
    """
    args_schema: Type[BaseModel] = GetXbrlDataInput

    def _run(self, htm_url: Optional[str] = None, xbrl_url: Optional[str] = None, accession_no: Optional[str] = None) -> str:
        """Use the tool."""
        if not (htm_url or xbrl_url or accession_no):
            error_msg = "Error: Must provide one of htm_url, xbrl_url, or accession_no."
            logger.error(error_msg)
            return error_msg
        if sum(p is not None for p in [htm_url, xbrl_url, accession_no]) > 1:
             error_msg = "Error: Provide only one of htm_url, xbrl_url, or accession_no."
             logger.error(error_msg)
             return error_msg

        try:
            params = {}
            if htm_url: params['htm_url'] = htm_url
            if xbrl_url: params['xbrl_url'] = xbrl_url
            if accession_no: params['accession_no'] = accession_no

            # print(f"\n>> Calling GetXbrlDataTool with params: {params}") // Replaced
            logger.info(f"GetXbrlDataTool called with params: {params}")
            xbrl_json = xbrl_api.xbrl_to_json(**params)
            # print(f">> Successfully converted XBRL to JSON.") // Replaced
            logger.info("Successfully converted XBRL to JSON.")

            # Selectively return key parts to manage context size
            relevant_data = {
                "CoverPage": xbrl_json.get("CoverPage", {}),
                "StatementsOfIncome": xbrl_json.get("StatementsOfIncome", {}),
                "BalanceSheets": xbrl_json.get("BalanceSheets", {}),
                "StatementsOfCashFlows": xbrl_json.get("StatementsOfCashFlows", {})
            }
            output_str = json.dumps(relevant_data, indent=2)
            output_len = len(output_str)
            limit = 20000 # Limit context length
            if output_len > limit:
                # print(f">> Truncating XBRL JSON output from {len(output_str)} to {limit} chars.") // Replaced
                logger.warning(f"Truncating XBRL JSON output from {output_len} to {limit} chars.")
                output_str = output_str[:limit] + "... (truncated JSON)}"
            logger.info(f"Returning XBRL data (length: {len(output_str)}).")
            return output_str

        except Exception as e:
            # print(f"\n>> Error in GetXbrlDataTool: {e}") // Replaced
            error_msg = f"Exception in GetXbrlDataTool: {e}. Params: {params}. Ensure the filing has XBRL data and the identifier is correct."
            logger.exception(error_msg)
            return f"Error executing get_xbrl_data_as_json: {e}. Ensure the filing has XBRL data and the identifier is correct."

# --- Define Custom Callback Handler for Detailed Logging ---

class SmartFileLoggerCallback(BaseCallbackHandler):
    """Enhanced callback handler with configurable logging levels"""

    def __init__(self, log_level: str = "INFO"): # Removed log_dir argument as it's handled by setup_logging
        """
        Initialize the callback handler. Assumes logging is already configured.

        Args:
            log_level (str): Logging level for messages handled by *this callback*
                             (DEBUG, INFO, etc.). Note: Overall logging level is
                             set in setup_logging.
        """
        # Remove logging configuration from here
        # os.makedirs(log_dir, exist_ok=True) # Handled by setup_logging
        self.log_level = getattr(logging, log_level.upper())
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%Z") # Handled by setup_logging
        # log_file = os.path.join(log_dir, f"log_{timestamp}.log") # Handled by setup_logging

        # Remove basicConfig call
        # logging.basicConfig(
        #     level=self.log_level,
        #     format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        #     handlers=[
        #         logging.FileHandler(log_file),
        #         logging.StreamHandler()
        #     ]
        # )

        # Just get the logger instance configured by setup_logging
        self.logger = logging.getLogger(__name__) # Or logging.getLogger() if preferred
        self.chain_depth = 0
        self.important_chains = {
            "AgentExecutor",
            "LLMChain",
            "SearchFilingsTool",
            "ExtractSectionTool"
        }
        # Log callback initialization if needed (optional)
        # self.logger.debug("SmartFileLoggerCallback initialized.")

    def _should_log_chain(self, serialized: Dict[str, Any]) -> bool:
        """Determine if this chain should be logged based on logging level"""
        # --- ADDED CHECK FOR NONE ---
        if not serialized:
             # Don't log if serialized is None
             return False
        # --- END CHECK ---
        chain_name = serialized.get('name', '<unknown_chain>')
        if self.log_level <= logging.DEBUG:
            return True
        return chain_name in self.important_chains

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Log chain starts with proper indentation and filtering"""
        # --- MOVED CHECK TO _should_log_chain, but keep depth increment ---
        should_log = self._should_log_chain(serialized)
        if should_log:
            # --- ADDED CHECK FOR NONE (redundant but safe) ---
            chain_name = serialized.get('name', '<unknown_chain>') if serialized else '<unknown_chain>'
            # --- END CHECK ---
            indent = "  " * self.chain_depth

            # Format inputs for better readability
            input_str = self._format_inputs(inputs)

            self.logger.log(
                logging.DEBUG if chain_name == '<unknown_chain>' else logging.INFO,
                f"{indent}Chain Start: {chain_name}\n{indent}Inputs: {input_str}"
            )
        # --- Always increment depth regardless of logging ---
        self.chain_depth += 1

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Log chain ends with proper indentation and filtering"""
        # --- Always decrement depth ---
        self.chain_depth -= 1
        # --- Check if we should log the end (based on debug level or if it's the outermost chain) ---
        # --- We need the serialized object here to check the name, let's get it from kwargs if available ---
        serialized = kwargs.get('serialized', None) # Attempt to get serialized info if passed
        should_log_end = self.log_level <= logging.DEBUG or self.chain_depth == 0
        # Optionally refine: only log end if start was logged (requires storing state, maybe too complex for now)

        if should_log_end:
            # --- Determine chain name for logging if possible ---
            chain_name = serialized.get('name', '<unknown_chain>') if serialized else '<unknown_chain>'
            indent = "  " * self.chain_depth
            output_str = self._format_outputs(outputs)
            # Log level INFO for outer chain end, DEBUG for inner ones if DEBUG level is set
            log_level = logging.INFO if self.chain_depth == 0 else logging.DEBUG
            if self.log_level <= log_level: # Only log if configured level allows
                 self.logger.log(log_level, f"{indent}Chain End ({chain_name}): {output_str}")


    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Log tool usage with detailed information"""
        # --- ADDED CHECK FOR NONE ---
        tool_name = serialized.get('name', '<unknown_tool>') if serialized else '<unknown_tool>'
        # --- END CHECK ---
        self.logger.info(f"Tool Start: {tool_name}\nInput: {input_str}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log tool completion with truncated output"""
        preview = output[:200] + "..." if len(output) > 200 else output
        self.logger.info(f"Tool End: {preview}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Log LLM calls with optional prompt logging"""
        if self.log_level <= logging.DEBUG:
            self.logger.debug(f"LLM Start: {serialized.get('name', 'unknown')}\nPrompts: {prompts}")
        else:
            self.logger.info("LLM Start: Processing request...")

    def on_llm_end(self, response, **kwargs) -> None:
        """Log LLM completion"""
        if self.log_level <= logging.DEBUG:
            self.logger.debug(f"LLM End: {response}")
        else:
            self.logger.info("LLM End: Response received")

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Log agent decisions using the AgentAction object."""
        tool_name = action.tool
        tool_input = action.tool_input
        # Format tool_input for better readability if it's a dict
        try:
            # Attempt to pretty-print if it's JSON-like
            input_str = json.dumps(tool_input, indent=2)
        except TypeError:
             input_str = str(tool_input) # Fallback to string representation

        self.logger.info(f"Agent Action: Tool='{tool_name}', Input='{input_str}'")
        # Log the reasoning if available in the action's log string
        if action.log and "Invoking" not in action.log: # Avoid redundant logging if reasoning is just "Invoking..."
             self.logger.info(f"Agent Reasoning:\n{action.log.strip()}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Log the final output when the agent finishes."""
        output = finish.return_values.get('output', 'N/A')
        # Format output for logging (similar to on_chain_end)
        output_str = self._format_outputs(finish.return_values)
        self.logger.info(f"Agent Finish: {output_str}")

    def _format_inputs(self, inputs: Dict[str, Any]) -> str:
        """Format input dictionary for readable logging"""
        # --- ADDED CHECK FOR NON-DICT INPUTS ---
        if not isinstance(inputs, dict):
            return str(inputs) # Return string representation if not a dict
        # --- END CHECK ---
        try:
            # --- ADDED HANDLING FOR AGENT_SCRATCHPAD ---
            formatted = {}
            for k, v in inputs.items():
                if k == 'agent_scratchpad':
                     # Represent scratchpad concisely
                     scratchpad_preview = repr(v)[:150] + "..." if len(repr(v)) > 150 else repr(v)
                     formatted[k] = f"<AgentScratchpad: {scratchpad_preview}>"
                elif isinstance(v, str) and len(v) > 300: # Slightly longer limit for inputs
                     formatted[k] = v[:300] + "..."
                else:
                     formatted[k] = v
            return json.dumps(formatted, indent=2)
            # --- END HANDLING ---
        except Exception: # Catch broader exceptions during formatting
            return str(inputs) # Fallback

    def _format_outputs(self, outputs: Dict[str, Any]) -> str:
        """Format output dictionary for readable logging"""
        # --- ADDED CHECK FOR NON-DICT OUTPUTS ---
        if not isinstance(outputs, dict):
            return str(outputs) # Return string representation if not a dict
        # --- END CHECK ---
        try:
            # Truncate long outputs
            formatted = {}
            for k, v in outputs.items():
                if isinstance(v, str) and len(v) > 200:
                    formatted[k] = v[:200] + "..."
                else:
                    formatted[k] = v
            return json.dumps(formatted, indent=2)
        except Exception: # Catch broader exceptions during formatting
            return str(outputs) # Fallback

    def on_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Log errors with full traceback in debug mode"""
        if self.log_level <= logging.DEBUG:
            self.logger.exception("Error occurred during execution:", exc_info=error)
        else:
            self.logger.error(f"Error: {str(error)}")

# --- Set up LLM and Agent ---
llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-flash-latest", # Switched from Flash
    model="gemini-1.5-pro-latest",   # Switched to Pro
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)
logger.info(f"Initializing LLM with model: {llm.model}") # More specific log

tools = [
    SearchFilingsTool(),
    ExtractSectionTool(),
    MapIdentifiersTool(),
    GetXbrlDataTool(),
]
logger.info("Tools initialized.") # Added specific log

# --- Enhanced System Prompt ---
system_prompt = """You are a specialized AI assistant for querying U.S. Securities and Exchange Commission (SEC) filings.
Your goal is to accurately answer user questions based *only* on data retrieved from SEC filings using the provided tools.

**Core Functionality:**
1.  **Understand the Query:** Determine the company (ticker/CIK), form type (10-K, 10-Q, 8-K, etc.), date range, and specific information requested (e.g., section, financial data point).
2.  **Map Identifiers (If Necessary):** If the user provides a company name or ticker and you need the CIK (or vice-versa) for searching, use the `map_sec_identifiers` tool.
3.  **Search for Filings:** Use the `search_sec_filings` tool to find the relevant filing(s).
    *   Pay close attention to dates. Use `filedAt` for filing dates and `periodOfReport` for the reporting period dates.
    *   **CRITICAL:** For requests like "latest" or "most recent", you **MUST** include `sort: [{{'field_name': 'filedAt', 'order': 'desc'}}]` and usually `size: 1` in the `search_sec_filings` parameters.
    *   **Note the output:** Pay close attention to the `linkToFilingDetails` URL and `accessionNo` returned by this tool.
4.  **Extract Information:**
    *   **Use Search Results:** You **MUST** use the `linkToFilingDetails` URL (or the primary `.htm` document URL if `linkToFilingDetails` is unavailable) provided in the output of the **immediately preceding** `search_sec_filings` call as the `filing_url` for `extract_filing_section`, or the `htm_url`/`accession_no` for `get_xbrl_data_as_json`. **Do not use URLs from memory or previous turns.**
    *   For specific sections (like '1A', '7', '8' in 10-K; 'part1item1', 'part2item1a' in 10-Q; '1-1', '5-2' in 8-K), use the `extract_filing_section` tool with the correct `filing_url` (from search results) and `section` identifier.
    *   **IMPORTANT:** If the user asks for specific information *within* a section (e.g., "What does the MD&A say about liquidity?", "Find cybersecurity risks in Item 1A"), provide the specific keywords or topic from the user's query as the `user_query` parameter to the `extract_filing_section` tool. This helps retrieve relevant chunks from long sections.
    *   For structured financial data (Revenue, Net Income, Assets, etc.), use the `get_xbrl_data_as_json` tool with the `htm_url` or `accession_no` obtained from the **immediately preceding** `search_sec_filings` call.
5.  **Synthesize the Answer:** Combine the information retrieved from the tools into a clear answer.
    *   **Formatting for Summaries:** When summarizing content extracted from sections like Risk Factors (1A, part2item1a), MD&A (7, part1item2), or Business (1), present the information as a **detailed bulleted list**. Identify the main categories or themes mentioned in the retrieved text and provide specific key points or examples under each. Avoid overly brief summaries; aim for informativeness suitable for analysis.
    *   For other queries (e.g., specific XBRL data points, identifier mappings), provide the direct information clearly.

**CRITICAL RULES:**
*   **MANDATORY TOOL USAGE:** You **MUST** use the provided tools (`search_sec_filings`, `extract_filing_section`, `map_sec_identifiers`, `get_xbrl_data_as_json`) to answer any questions about SEC filings, company data (like CIK, tickers), specific filing contents, sections, or financial data.
*   **NO INTERNAL KNOWLEDGE FOR FILING DATA:** You **MUST NOT** invent, guess, or retrieve from your internal knowledge any specific details about SEC filings, including URLs, dates, section contents, or financial figures. Rely **exclusively** on the output of the tools.
*   **USE TOOL OUTPUT DIRECTLY:** When chaining tool calls (e.g., search then extract), you **MUST** use the relevant data (like URLs or IDs) provided in the output of the preceding tool call as input for the next tool call.
*   **REPORT FAILURES:** If a tool fails or does not return the necessary information after diligent use, you **MUST** state that the information could not be found using the available tools, specifying which tool failed and why if possible (based on the error message). Do not attempt to generate an answer based on assumptions or prior knowledge.
*   **SOURCE ATTRIBUTION:** In your final answer, **ALWAYS** include a brief summary of the main actions taken (e.g., "Searched for AAPL 10-K", "Extracted Section 1A") and **ALWAYS** cite the specific `filing_url` or `accession_no` of the document(s) used as the source.

**Section Identifiers for `extract_filing_section`:**
*   **10-K:** '1', '1A', '1B', '1C', '2', '3', '4', '5', '6', '7', '7A', '8', '9', '9A', '9B', '10', '11', '12', '13', '14' # Added 1C, removed 15
*   **10-Q:** 'part1item1', 'part1item2', 'part1item3', 'part1item4', 'part2item1', 'part2item1a', 'part2item2', 'part2item3', 'part2item4', 'part2item5', 'part2item6'
*   **8-K:** Use item numbers like '1-1', '1-2', '2-2', '5-2', '8-1', etc. (Refer to 8-K item list if unsure).

**Example Final Answer Format (for Summary):**
"Based on searching for [Company]'s [Form Type] (using `search_sec_filings`) and extracting [Section Name] (using `extract_filing_section`) from the filing dated [Date] (Source: [filing_url]), here is a detailed summary of the key points:\n\n*   **[Risk Category 1]:**\n    *   [Specific point/example from text]\n    *   [Another specific point/example]\n*   **[Risk Category 2]:**\n    *   [Specific point/example]\n*   **[Other Key Points]:**\n    *   [Specific point]"

Now, answer the user's question following these instructions precisely.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
logger.info("Prompt template created.")

# Get log level from environment variable or default to INFO
log_level = os.getenv("SEC_AGENT_LOG_LEVEL", "INFO")

# Initialize the callback handler (No log_dir needed here anymore)
callback = SmartFileLoggerCallback(
    log_level=log_level
)

# Create your agent with the callback
agent = create_tool_calling_agent(llm, tools, prompt)
logger.info("Agent created.") # Added specific log

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Keep console verbose output for comparison if needed
    handle_parsing_errors=True,
    # Return intermediate steps allows inspection later if needed, but callbacks are primary
    return_intermediate_steps=False
    )

logger.info("AgentExecutor created.")

# --- Main Interaction Loop ---
if __name__ == "__main__":
    # Use the callback instance created earlier
    callbacks_list = [callback]

    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Join arguments (excluding script name) to form the query
        user_query = " ".join(sys.argv[1:])
        logger.info(f"Received user query from command-line arguments: {user_query}")
        try:
            # Execute the agent once with the provided query AND the callback
            response = agent_executor.invoke(
                {"input": user_query},
                config={"callbacks": callbacks_list} # Pass callbacks here
            )
            # Log final response output AFTER agent execution finishes
            logger.info(f"Agent final response output: {response.get('output', 'N/A')}")
            # Print the final response to the console
            print(f"\nAssistant:\n{response.get('output', 'Error: No output found.')}")
        except Exception as e:
            # Log exception using the logger, which will include traceback via callback
            # logger.exception(f"An error occurred during agent execution for command-line query: {e}")
            # The callback's on_chain_error should handle logging the exception
            print(f"\nAssistant: An error occurred: {e}") # Also inform user on console
    else:
        # No command-line arguments, start the interactive loop
        logger.info("No command-line query provided. Starting main interaction loop.")
        while True:
            try:
                user_query = input("\nYour Question: ")
            except EOFError: # Handle EOF if input is piped
                 logger.info("EOF received, exiting.")
                 break
            if user_query.lower() == 'quit':
                logger.info("Quit command received, exiting.")
                break
            if not user_query:
                continue

            logger.info(f"Received user query from input: {user_query}")
            try:
                # Use invoke for synchronous execution WITH the callback
                response = agent_executor.invoke(
                    {"input": user_query},
                    config={"callbacks": callbacks_list} # Pass callbacks here
                )
                # Log final response output AFTER agent execution finishes
                logger.info(f"Agent final response output: {response.get('output', 'N/A')}")
                # AgentExecutor with verbose=True already prints steps; avoid double printing response.
                # If verbose=False, uncomment the print below:
                print(f"\nAssistant:\n{response.get('output', 'Error: No output found.')}")

            except Exception as e:
                # Log exception using the logger, which will include traceback via callback
                # logger.exception(f"An error occurred during agent execution: {e}")
                # The callback's on_chain_error should handle logging the exception
                print(f"\nAssistant: An error occurred: {e}") # Also inform user on console

        logger.info("Exited main execution.")