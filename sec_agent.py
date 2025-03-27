import os
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Type
from enum import Enum # Import Enum
import logging # <-- Add logging
import datetime # <-- Add datetime
import pytz # <-- Add pytz
import sys # <-- Import sys to handle command-line arguments

# Use Pydantic v2 directly
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

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

# --- Define Tool Input Schemas using Pydantic v2 ---

# Define Enum for sort order
class SortOrderEnum(str, Enum):
    asc = "asc"
    desc = "desc"

# Define a clear structure for a single sort criterion
class SortCriterion(BaseModel):
    field_name: str = Field(description="The name of the field to sort by (e.g., 'filedAt', 'periodOfReport').")
    order: SortOrderEnum = Field(description="The sort order: 'asc' for ascending, 'desc' for descending.")

class SearchFilingsInput(BaseModel):
    query: str = Field(description="Mandatory. The search query string using SEC-API query syntax (similar to Lucene). Examples: 'ticker:AAPL AND formType:\"10-K\"', 'formType:\"8-K\" AND items:\"1.01\" AND filedAt:[2023-01-01 TO 2023-12-31]'. Use specific fields like ticker, cik, formType, filedAt, periodOfReport, items, sic, etc.")
    start: int = Field(0, description="Optional. The starting index for results (pagination). Default is 0.")
    size: int = Field(10, description="Optional. The number of results to return (pagination). Default is 10. Max is typically 200, check sec-api docs.")
    # Use the simplified List[SortCriterion] structure
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
    description: str = """
    Searches and filters SEC EDGAR filings based on various criteria like ticker, CIK, company name,
    form type (e.g., "10-K", "10-Q", "8-K", "13F-HR"), filing date, reporting period, SIC code, 8-K items, etc.
    Uses the SEC-API query syntax (like Lucene). Returns metadata about matching filings, including links.
    Crucial for finding relevant filings before extracting data.
    Use `filedAt:[YYYY-MM-DD TO YYYY-MM-DD]` for date ranges.
    Use `periodOfReport:[YYYY-MM-DD TO YYYY-MM-DD]` for report period ranges.
    Combine criteria using AND/OR. Example: 'ticker:MSFT AND formType:"10-K" AND filedAt:[2023-01-01 TO 2023-12-31]'
    Sorting might be limited to fields like 'filedAt'. Check sec-api documentation if unsure.
    """
    args_schema: Type[BaseModel] = SearchFilingsInput

    def _run(self, query: str, start: int = 0, size: int = 10, sort: Optional[List[SortCriterion]] = None) -> str:
        """Use the tool."""
        try:
            search_query = {
                "query": query,
                "from": str(start),
                "size": str(size),
            }
            if sort:
                # Ensure sort criteria are serializable for logging
                try:
                    sort_repr = json.dumps([criterion.dict() for criterion in sort])
                except Exception:
                    sort_repr = "[unserializable sort object]"
                api_sort = [{criterion.field_name: {"order": criterion.order.value}} for criterion in sort]
                search_query["sort"] = api_sort
                logger.info(f"SearchFilingsTool called with query: {query}, start: {start}, size: {size}, sort: {sort_repr}")
            else:
                search_query["sort"] = [{"filedAt": {"order": "desc"}}] # Default sort
                logger.info(f"SearchFilingsTool called with query: {query}, start: {start}, size: {size}, sort: Default (filedAt desc)")

            # print(f"\n>> Calling SearchFilingsTool with query: {search_query}") // Replaced
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
    Requires the direct URL to the filing's HTML document and the correct section identifier (e.g., '1A', 'part1item1', '5-2').
    Use this *after* finding the relevant filing URL with 'search_sec_filings'.
    If the section is very long, it attempts to retrieve chunks relevant to the user's query.
    Returns the cleaned text content of the specified section or relevant excerpts.
    """
    args_schema: Type[BaseModel] = ExtractSectionInput

    def _run(self, filing_url: str, section: str, return_type: str = "text", user_query: Optional[str] = None) -> str:
        """Use the tool. Implements basic RAG for long sections."""
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
                retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks
                retrieval_query = user_query # Use the user query passed into the tool

                if not retrieval_query:
                    logger.warning("No user_query provided for RAG retrieval. Using section name as fallback query.")
                    # Basic fallback: use section name or generic term
                    section_map = { "1A": "risk factors", "7": "management discussion analysis", "8": "financial statements footnotes", "part1item1": "financial statements footnotes", "part1item2": "management discussion analysis", "part2item1a": "risk factors", "1C": "cybersecurity"}
                    retrieval_query = section_map.get(section, section) # Use section ID if not in map

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


# --- Set up LLM and Agent ---
llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro-latest",
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)
logger.info("Initializing LLM.") # Added specific log

tools = [
    SearchFilingsTool(),
    ExtractSectionTool(),
    MapIdentifiersTool(),
    GetXbrlDataTool(),
]
logger.info("Tools initialized.") # Added specific log

# Updated prompt with more specific guidance for section extraction
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful financial analyst assistant specializing in SEC filings.
You have access to tools that can search SEC filings, extract specific sections from filings,
map company identifiers (like ticker to CIK), and extract structured XBRL financial data.

**Tool Usage Strategy:**

1.  **Identify Goal:** Understand what specific information the user needs (e.g., risk factors, revenue policy, specific financial number, acquisition details, executive changes).
2.  **Find Filing(s):** Use `search_sec_filings`. Be specific with `query` parameters (ticker, formType, dates). Default sort is `filedAt` descending (latest first). **Determine the `formType` (e.g., "10-K", "10-Q", "8-K") from the search results.**
3.  **Extract Information:** Use `extract_filing_section` with the `filing_url` from the search results. **Choose the `section` parameter carefully based on the `formType` of the filing and the information needed, using the exact identifiers listed below.**
    *   **IMPORTANT:** If the section might be long and you need specific details, **pass the original user query to the `user_query` parameter** of the `extract_filing_section` tool. This helps retrieve the most relevant parts.

    **Supported Sections for `extract_filing_section`:**

    *   **If `formType` is "10-K":**
        *   `'1'` - Business
        *   `'1A'` - Risk Factors
        *   `'1B'` - Unresolved Staff Comments
        *   `'1C'` - Cybersecurity (introduced in 2023)
        *   `'2'` - Properties
        *   `'3'` - Legal Proceedings
        *   `'4'` - Mine Safety Disclosures
        *   `'5'` - Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities
        *   `'6'` - Selected Financial Data (prior to Feb 2021)
        *   `'7'` - Management's Discussion and Analysis of Financial Condition and Results of Operations (MD&A)
        *   `'7A'` - Quantitative and Qualitative Disclosures about Market Risk
        *   `'8'` - Financial Statements and Supplementary Data (Includes Footnotes)
        *   `'9'` - Changes in and Disagreements with Accountants on Accounting and Financial Disclosure
        *   `'9A'` - Controls and Procedures
        *   `'9B'` - Other Information
        *   `'10'` - Directors, Executive Officers and Corporate Governance
        *   `'11'` - Executive Compensation
        *   `'12'` - Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters
        *   `'13'` - Certain Relationships and Related Transactions, and Director Independence
        *   `'14'` - Principal Accountant Fees and Services

    *   **If `formType` is "10-Q":**
        *   **Part 1:**
            *   `'part1item1'` - Financial Statements (Includes Footnotes)
            *   `'part1item2'` - Management's Discussion and Analysis of Financial Condition and Results of Operations (MD&A)
            *   `'part1item3'` - Quantitative and Qualitative Disclosures About Market Risk
            *   `'part1item4'` - Controls and Procedures
        *   **Part 2:**
            *   `'part2item1'` - Legal Proceedings
            *   `'part2item1a'` - Risk Factors
            *   `'part2item2'` - Unregistered Sales of Equity Securities and Use of Proceeds
            *   `'part2item3'` - Defaults Upon Senior Securities
            *   `'part2item4'` - Mine Safety Disclosures
            *   `'part2item5'` - Other Information
            *   `'part2item6'` - Exhibits

    *   **If `formType` is "8-K":** Use the item number with a hyphen, e.g.:
        *   `'1-1'` - Item 1.01 Entry into a Material Definitive Agreement
        *   `'1-2'` - Item 1.02 Termination of a Material Definitive Agreement
        *   `'2-2'` - Item 2.02 Results of Operations and Financial Condition
        *   `'4-1'` - Item 4.01 Changes in Registrant's Certifying Accountant
        *   `'4-2'` - Item 4.02 Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review
        *   `'5-2'` - Item 5.02 Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers
        *   `'8-1'` - Item 8.01 Other Events
        *   `'9-1'` - Item 9.01 Financial Statements and Exhibits
        *   *(Refer to SEC 8-K item list for all possible identifiers like 'X-Y')*

    *   **Note:** When looking for specific details like Accounting Policies, Revenue Recognition, or Cybersecurity disclosures within financial statements, extract the main financial statement section (`'8'` for 10-K, `'part1item1'` for 10-Q) and then analyze the returned text. For 10-K filings after 2023, also consider extracting section `'1C'` specifically for Cybersecurity.

4.  **XBRL Data:** If the user asks for specific, quantifiable financial data (e.g., "What was the exact revenue?", "How much inventory?"), consider using `get_xbrl_data_as_json` *after* finding the filing. This provides structured data if available.
5.  **Synthesize Answer:** Base your final answer *only* on the information retrieved by the tools. If a tool fails or the information isn't found in the extracted sections, state that clearly. Do not make assumptions.

**Example Workflow (10-Q Risk Factors):**
User: "What are the latest risk factors mentioned in MSFT's 10-Q?"
1. `search_sec_filings` (query='ticker:MSFT AND formType:"10-Q"', size=1) -> Get URL and confirm formType is "10-Q".
2. `extract_filing_section` (filing_url=URL, section='part2item1a') -> Get text of Risk Factors for the 10-Q.
3. Formulate answer based on the extracted text.
"""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
logger.info("Prompt template created.") # Added specific log

agent = create_tool_calling_agent(llm, tools, prompt)
logger.info("Agent created.") # Added specific log

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Keep LangChain's verbose output for agent steps
    handle_parsing_errors=True # Helps agent recover from LLM output errors
    )

logger.info("AgentExecutor created.")

# --- Main Interaction Loop ---
if __name__ == "__main__":
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Join arguments (excluding script name) to form the query
        user_query = " ".join(sys.argv[1:])
        logger.info(f"Received user query from command-line arguments: {user_query}")
        try:
            # Execute the agent once with the provided query
            response = agent_executor.invoke({"input": user_query})
            logger.info(f"Agent final response output: {response.get('output', 'N/A')}")
            # Print the final response to the console
            print(f"\nAssistant:\n{response.get('output', 'Error: No output found.')}")
        except Exception as e:
            logger.exception(f"An error occurred during agent execution for command-line query: {e}")
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

            logger.info(f"Received user query from input: {user_query}") # Clarified source
            try:
                # Use invoke for synchronous execution
                response = agent_executor.invoke({"input": user_query})
                logger.info(f"Agent final response output: {response.get('output', 'N/A')}")
                # AgentExecutor with verbose=True already prints steps; avoid double printing response.
                # If verbose=False, uncomment the print below:
                print(f"\nAssistant:\n{response.get('output', 'Error: No output found.')}")

            except Exception as e:
                # Catch broader exceptions during agent execution
                logger.exception(f"An error occurred during agent execution: {e}")
                print(f"\nAssistant: An error occurred: {e}") # Also inform user on console

        logger.info("Exited main execution.") # Updated exit message