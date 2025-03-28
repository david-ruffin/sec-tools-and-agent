"""
This module provides functionality for analyzing SEC filings using SEC-API.io.
Following roadmap tasks 1.1 and 1.2 for API integration foundation.
"""

from sec_api import QueryApi, ExtractorApi, XbrlApi, FullTextSearchApi, RenderApi
import logging
import os
from typing import Dict, Optional, Any, Union, List
from datetime import datetime
import re
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextStore:
    """Simple text store for SEC filing chunks with Contextual RAG capabilities."""
    
    def __init__(self):
        """Initialize the text store."""
        self.chunks = []  # Store chunk data
        self.chunk_hashes = set()  # Track unique chunks
        
    def _generate_chunk_context(self, chunk: Dict[str, Any], filing_metadata: Dict[str, Any]) -> str:
        """
        Generate contextual information for a chunk based on its content and metadata.
        Following Anthropic's Contextual RAG approach.
        
        Args:
            chunk: The chunk dictionary containing text and metadata
            filing_metadata: Metadata about the filing
            
        Returns:
            Contextual information to prepend to the chunk
        """
        context_parts = []
        
        # Add filing context
        if filing_metadata.get("form"):
            context_parts.append(f"From {filing_metadata['form']} filing")
        if filing_metadata.get("periodOfReport"):
            context_parts.append(f"for period ending {filing_metadata['periodOfReport']}")
        if filing_metadata.get("companyName"):
            context_parts.append(f"of {filing_metadata['companyName']}")
            
        # Add section context
        if chunk.get("metadata", {}).get("section"):
            context_parts.append(f"in section {chunk['metadata']['section']}")
            
        # Add surrounding context
        if chunk.get("context_before"):
            context_parts.append(f"following: {chunk['context_before']}")
        if chunk.get("context_after"):
            context_parts.append(f"preceding: {chunk['context_after']}")
            
        # Add special content markers
        if chunk.get("metadata", {}).get("contains_table"):
            context_parts.append("contains tabular financial data")
        if any(marker in chunk.get("text", "") for marker in ["$", "%", "million", "billion"]):
            context_parts.append("contains numerical financial information")
            
        return "; ".join(context_parts)
        
    def add_chunks(self, chunks: List[Dict[str, Any]], filing_metadata: Dict[str, Any]) -> None:
        """
        Add chunks to the text store with contextual information.
        
        Args:
            chunks: List of chunk dictionaries
            filing_metadata: Metadata about the filing
        """
        for chunk in chunks:
            # Create unique hash of chunk content
            chunk_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            if chunk_hash in self.chunk_hashes:
                continue
                
            # Generate and add context
            context = self._generate_chunk_context(chunk, filing_metadata)
            chunk["context"] = context
            chunk["contextualized_text"] = f"{context}. {chunk['text']}"
            
            # Add filing metadata
            chunk["filing_metadata"] = filing_metadata
            chunk["chunk_hash"] = chunk_hash
            
            self.chunks.append(chunk)
            self.chunk_hashes.add(chunk_hash)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks using both contextual and content matching.
        Implements hybrid search combining BM25-style matching with contextual relevance.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of chunks with relevance scores
        """
        query_terms = set(query.lower().split())
        results = []
        
        for chunk in self.chunks:
            # Search in both context and content
            search_text = chunk["contextualized_text"].lower()
            
            # Calculate TF-IDF style score
            term_freq = sum(search_text.count(term) for term in query_terms)
            
            # Boost score based on context matches
            context_matches = sum(term in chunk["context"].lower() for term in query_terms)
            context_boost = 1 + (context_matches * 0.5)  # 50% boost per context match
            
            # Calculate final score
            score = term_freq * context_boost
            
            if score > 0:
                result = chunk.copy()
                result["relevance_score"] = score
                results.append(result)
        
        # Sort by relevance score and return top k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:k]
    
    def save(self, directory: str) -> None:
        """
        Save text store to disk.
        
        Args:
            directory: Directory to save files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save chunks and metadata
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump({
                "chunks": self.chunks,
                "chunk_hashes": list(self.chunk_hashes)
            }, f)
    
    def load(self, directory: str) -> None:
        """
        Load text store from disk.
        
        Args:
            directory: Directory containing saved files
        """
        # Load chunks and metadata
        with open(os.path.join(directory, "chunks.json"), "r") as f:
            data = json.load(f)
            self.chunks = data["chunks"]
            self.chunk_hashes = set(data["chunk_hashes"])

class SecApiError(Exception):
    """Custom exception for SEC API errors."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class SecApiIntegration:
    """A class to handle integration with the SEC API."""
    
    ERROR_MESSAGES = {
        401: "Please check your SEC API key or request a new one",
        403: "Access denied. Please verify your API permissions",
        404: "No filing found matching the specified criteria",
        429: "Please wait before making additional requests",
        500: "SEC API internal error. Please try again later",
        503: "SEC API service unavailable. Please try again later"
    }
    
    # Section markers for chunking
    SECTION_MARKERS = [
        r"Item \d+[A-Z]?\.",  # Item 1., Item 1A., etc.
        r"\n[A-Z][^\n]+\n[-=]+\n",  # Headers with underlines
        r"\n\d+\.\s+[A-Z][^\n]+\n",  # Numbered sections
        r"\n[A-Z][A-Z\s]+\n"  # ALL CAPS headers
    ]
    
    def __init__(self, sec_api_key: Optional[str] = None):
        """Initialize the SEC API integration with required API clients."""
        # Clear any existing environment variable to ensure proper testing
        if 'SEC_API_KEY' in os.environ:
            del os.environ['SEC_API_KEY']
            
        api_key = sec_api_key or os.getenv('SEC_API_KEY')
        if not api_key:
            raise ValueError('SEC API key is required')
            
        self.sec_api_key = api_key
        
        # Initialize SEC API clients
        self.query_api = QueryApi(api_key=self.sec_api_key)
        self.extractor_api = ExtractorApi(api_key=self.sec_api_key)
        self.xbrl_api = XbrlApi(api_key=self.sec_api_key)
        self.full_text_api = FullTextSearchApi(api_key=self.sec_api_key)
        self.render_api = RenderApi(api_key=self.sec_api_key)
        
        # Initialize text store instead of vector store
        self.text_store = TextStore()
    
    def search_filings(self, query: Union[Dict[str, Any], str], form_type: Optional[str] = None, 
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for SEC filings using the Query API.
        
        Args:
            query: Either a complete query dictionary or a ticker symbol
            form_type: Optional form type (e.g., '10-K', '10-Q')
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            
        Returns:
            Dict containing filing data or error message
        """
        try:
            if isinstance(query, str):
                # Build query from parameters
                query_dict = {
                    "query": f"ticker:{query}" + 
                            (f" AND formType:\"{form_type}\"" if form_type else "") +
                            (f" AND filedAt:[{start_date} TO {end_date}]" if start_date and end_date else ""),
                    "from": "0",
                    "size": "10",
                    "sort": [{"filedAt": {"order": "desc"}}]
                }
            else:
                # Convert from_ to from in query parameters
                query_dict = {k.replace('from_', 'from'): v for k, v in query.items()}
            
            filings = self.query_api.get_filings(query_dict)
            
            # Validate each filing's metadata
            valid_filings = []
            for filing in filings.get("filings", []):
                # Add required fields if missing but can be inferred
                if "linkToFilingDetails" not in filing and "accessionNo" in filing:
                    filing["linkToFilingDetails"] = f"https://www.sec.gov/Archives/edgar/data/{filing['accessionNo']}"
                if "periodOfReport" not in filing:
                    filing["periodOfReport"] = filing.get("filedAt", "")  # Use filedAt as fallback
                
                if self.validate_filing_metadata(filing):
                    valid_filings.append(filing)
                else:
                    logger.warning(f"Skipping invalid filing metadata: {filing}")
            
            return {
                "filings": valid_filings,
                "total": len(valid_filings)
            }
            
        except Exception as e:
            error_data = str(e)
            error_msg = self._get_error_message(error_data)
            logger.error(f"Error searching filings: {error_data}")
            return {"error": f"Error: {error_msg}", "filings": []}

    def extract_section(self, filing_url: str, section: str, output_format: str = "text") -> Union[str, Dict[str, Any]]:
        """
        Extract a specific section from a filing using the Extractor API.
        
        Args:
            filing_url: URL of the filing
            section: Section identifier (e.g., "1A" for Risk Factors)
            output_format: Format of the output ("text" or "html")
            
        Returns:
            String content if successful, Dict with error info if failed
        """
        try:
            # Get section content
            section_data = self.extractor_api.get_section(filing_url, section, output_format)
            if not section_data:
                return "Section not found or empty"

            # Clean and validate content
            if output_format == "text":
                # Ensure section header is present
                expected_header = f"Item {section}."
                if expected_header not in section_data[:500]:  # Check first 500 chars
                    logger.warning(f"Section header '{expected_header}' not found in content")
                
                # Verify subsection structure (for Business and Risk Factors)
                if section in ["1", "1A"]:
                    subsections = section_data.split("\n\n")  # Split by double newline
                    if len(subsections) < 3:  # Should have at least header, overview, and one subsection
                        logger.warning("Section may be missing subsections")
                
                # Check for potential truncation
                if len(section_data) >= 100000:  # Large sections need special handling
                    logger.info(f"Large section detected ({len(section_data)} chars)")
                    # Verify the section has a proper ending (period, newline, etc.)
                    if not section_data.rstrip().endswith((".", "}", "]", ")", "\n")):
                        logger.warning("Section may be truncated")
            
            return section_data
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error extracting section: {str(e)}")
            return error_msg

    def get_financial_data(self, filing_url: str) -> Dict[str, Any]:
        """
        Get XBRL financial data using the XBRL API.
        
        Args:
            filing_url: URL of the filing
            
        Returns:
            Dict containing financial data or error message
        """
        try:
            financial_data = self.xbrl_api.xbrl_to_json(filing_url)
            if not financial_data:
                return {}

            # Extract and validate core financial statements
            processed_data = {
                "income_statement": self._process_income_statement(financial_data.get("IncomeStatement", {})),
                "balance_sheet": self._process_balance_sheet(financial_data.get("BalanceSheet", {})),
                "cash_flow": self._process_cash_flow(financial_data.get("CashFlow", {})),
                "shares": self._process_shares_data(financial_data.get("SharesOutstanding", {}))
            }
            
            return {
                "raw_data": financial_data,
                **processed_data
            }
        except Exception as e:
            error_msg = f"Error getting financial data: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _process_income_statement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate income statement data."""
        if not data:
            return {}
            
        result = {
            "Revenue": data.get("Revenue", 0),
            "context": {
                "period": data.get("context", {}).get("period", ""),
                "units": data.get("context", {}).get("units", "USD"),
                "duration": data.get("context", {}).get("duration", "")
            }
        }
        
        # Validate revenue data
        if result["Revenue"] is None:
            logger.warning("Revenue data is missing")
            result["Revenue"] = 0
            
        # Validate period format
        period = result["context"]["period"]
        try:
            if "-Q" in period:  # Quarterly format (e.g., "2024-Q3")
                year, quarter = period.split("-Q")
                assert 2000 <= int(year) <= 2100
                assert 1 <= int(quarter) <= 4
            else:  # Date format
                datetime.strptime(period, "%Y-%m-%d")
        except (ValueError, AssertionError):
            logger.error(f"Invalid period format: {period}")
            result["context"]["period"] = ""
            
        return result

    def _process_balance_sheet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate balance sheet data."""
        if not data:
            return {}
            
        return {
            "Assets": data.get("Assets", 0),
            "Liabilities": data.get("Liabilities", 0),
            "StockholdersEquity": data.get("StockholdersEquity", 0),
            "context": data.get("context", {})
        }

    def _process_cash_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate cash flow data."""
        if not data:
            return {}
            
        return {
            "CashFromOperations": data.get("CashFromOperations", 0),
            "InvestingCashFlow": data.get("InvestingCashFlow", 0),
            "FinancingCashFlow": data.get("FinancingCashFlow", 0),
            "context": data.get("context", {})
        }

    def _process_shares_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate shares outstanding data."""
        if not data:
            return {}
            
        result = {
            "CommonStock": data.get("CommonStock", 0),
            "PreferredStock": data.get("PreferredStock", 0),
            "context": {
                "asOf": data.get("context", {}).get("asOf", ""),
                "units": data.get("context", {}).get("units", "shares")
            }
        }
        
        # Validate date format
        try:
            if result["context"]["asOf"]:
                datetime.strptime(result["context"]["asOf"], "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format in shares data: {result['context']['asOf']}")
            result["context"]["asOf"] = ""
            
        return result

    def full_text_search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform full-text search using the Full-Text Search API.
        
        Args:
            query: Dictionary containing search parameters
            
        Returns:
            Dict containing search results or error message
        """
        try:
            search_results = self.full_text_api.get_filings(query)
            if not search_results:
                return {"filings": [], "total": 0, "error": "No results found"}
                
            # Ensure we return just the filings array and total
            if isinstance(search_results, dict):
                filings = search_results.get("filings", [])
                total = search_results.get("total", len(filings))
                return {"filings": filings, "total": total}
            
            # If search_results is a list, treat it as filings
            return {"filings": search_results, "total": len(search_results)}
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error performing full-text search: {str(e)}")
            return {"error": error_msg, "filings": [], "total": 0}

    def validate_filing_metadata(self, filing: Dict[str, Any]) -> bool:
        """
        Validate filing metadata structure and required fields.
        
        Args:
            filing: Filing metadata dictionary
            
        Returns:
            bool: True if metadata is valid, False otherwise
        """
        required_fields = [
            "accessionNo",
            "filedAt",
            "form",
            "periodOfReport",
            "linkToFilingDetails"
        ]
        
        # Check required fields
        if not all(field in filing for field in required_fields):
            logger.error(f"Missing required fields in filing metadata: {filing}")
            return False
        
        # Validate date formats
        try:
            filed_at = datetime.strptime(filing["filedAt"][:10], "%Y-%m-%d")
            period_of_report = datetime.strptime(filing["periodOfReport"][:10], "%Y-%m-%d")
            
            # Period of report should be before or equal to filed date
            if period_of_report.date() > filed_at.date():
                logger.error(f"Invalid dates: periodOfReport {period_of_report} is after filedAt {filed_at}")
                return False
        except ValueError as e:
            logger.error(f"Invalid date format in filing metadata: {e}")
            return False
        
        return True
        
    def _get_error_message(self, error_data: str) -> str:
        """Get a user-friendly error message based on the error data."""
        try:
            # Try to parse error data for status code
            if "status" in error_data:
                import json
                error_dict = json.loads(error_data.replace("'", '"'))
                status_code = error_dict.get("status")
                if status_code in self.ERROR_MESSAGES:
                    return self.ERROR_MESSAGES[status_code]
        except:
            pass
            
        # Return original error if we can't parse it
        return error_data

    def chunk_section(self, section_text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Chunk section content intelligently for RAG processing.
        
        Args:
            section_text: The text content to chunk
            max_chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing chunks and their metadata
        """
        if not section_text:
            return []

        # Find all section boundaries
        boundaries = []
        for pattern in self.SECTION_MARKERS:
            for match in re.finditer(pattern, section_text):
                boundaries.append(match.start())
        boundaries.sort()

        # Add start and end positions
        boundaries = [0] + boundaries + [len(section_text)]

        chunks = []
        current_pos = 0
        
        while current_pos < len(section_text):
            # Find next boundary within max_chunk_size
            chunk_end = current_pos + max_chunk_size
            best_boundary = chunk_end
            
            # Look for natural boundaries
            for boundary in boundaries:
                if current_pos < boundary <= chunk_end:
                    best_boundary = boundary
                    break
            
            # Extract chunk
            chunk_text = section_text[current_pos:best_boundary]
            
            # Get context (previous and next sentences)
            context_before = ""
            if current_pos > 0:
                prev_text = section_text[max(0, current_pos-200):current_pos]
                last_sentence = re.split(r'[.!?]\s+', prev_text)[-1]
                context_before = last_sentence
            
            context_after = ""
            next_text = section_text[best_boundary:min(len(section_text), best_boundary+200)]
            if next_text:
                next_sentence = re.split(r'[.!?]\s+', next_text)[0]
                context_after = next_sentence
            
            # Create chunk with metadata
            chunk = {
                "text": chunk_text,
                "start_pos": current_pos,
                "end_pos": best_boundary,
                "context_before": context_before,
                "context_after": context_after,
                "metadata": {
                    "section_markers": [m for m in re.finditer("|".join(self.SECTION_MARKERS), chunk_text)],
                    "contains_table": bool(re.search(r"\|\s*[-]+\s*\|", chunk_text)),
                    "contains_list": bool(re.search(r"^\s*[-•*]\s+", chunk_text, re.MULTILINE))
                }
            }
            chunks.append(chunk)
            
            # Move position for next chunk, accounting for overlap
            current_pos = best_boundary - overlap
            if current_pos <= current_pos - overlap:  # Prevent infinite loop
                current_pos = best_boundary
        
        return chunks

    def extract_section_with_chunks(self, filing_url: str, section: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Extract a section and return it with intelligent chunks for RAG.
        
        Args:
            filing_url: URL of the filing
            section: Section identifier (e.g., "1A" for Risk Factors)
            output_format: Format of the output ("text" or "html")
            
        Returns:
            Dict containing section content, chunks, and metadata
        """
        try:
            # Get section content
            section_data = self.extract_section(filing_url, section, output_format)
            if isinstance(section_data, str) and "Error:" in section_data:
                return {"error": section_data}
            
            # Create chunks with overlap
            chunks = self.chunk_section(section_data)
            
            return {
                "content": section_data,
                "chunks": chunks,
                "metadata": {
                    "total_chunks": len(chunks),
                    "section": section,
                    "filing_url": filing_url
                }
            }
            
        except Exception as e:
            error_msg = f"Error chunking section: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def process_filing_for_rag(self, filing_url: str, sections: List[str]) -> Dict[str, Any]:
        """
        Process a filing for RAG by extracting sections and adding to vector store.
        
        Args:
            filing_url: URL of the filing
            sections: List of sections to extract (e.g., ["1", "1A", "7"])
            
        Returns:
            Dict containing processing results and stats
        """
        results = {
            "processed_sections": [],
            "failed_sections": [],
            "total_chunks": 0,
            "filing_url": filing_url
        }
        
        try:
            # Get filing metadata
            filing_metadata = {
                "url": filing_url,
                "processed_at": datetime.now().isoformat()
            }
            
            # Process each section
            for section in sections:
                try:
                    # Extract section with chunks
                    section_data = self.extract_section_with_chunks(filing_url, section)
                    
                    if "error" in section_data:
                        results["failed_sections"].append({
                            "section": section,
                            "error": section_data["error"]
                        })
                        continue
                    
                    # Add chunks to vector store
                    self.text_store.add_chunks(
                        section_data["chunks"],
                        filing_metadata
                    )
                    
                    # Update results
                    results["processed_sections"].append({
                        "section": section,
                        "num_chunks": len(section_data["chunks"])
                    })
                    results["total_chunks"] += len(section_data["chunks"])
                    
                except Exception as e:
                    logger.error(f"Error processing section {section}: {str(e)}")
                    results["failed_sections"].append({
                        "section": section,
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing filing: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using text-based search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of chunks with relevance scores
        """
        return self.text_store.search(query, k)

def analyze_sec_filing(query: str, filing_metadata: dict, sec_api_key: str, openai_api_key: str = None) -> str:
    """
    Analyze SEC filing based on a query. This is the main interface used by main.py.
    """
    # Initialize API
    api = SecApiIntegration(sec_api_key)
    
    # Search for the filing using company name and year
    filings = api.search_filings(
        filing_metadata['company_name'],  # Use actual company name
        filing_metadata['form_type'],     # Use form type from metadata
        start_date=f"{filing_metadata['year']}-01-01",
        end_date=f"{filing_metadata['year']}-12-31"
    )
    
    if not filings.get("filings"):
        return f"No {filing_metadata['form_type']} filings found for {filing_metadata['company_name']} in {filing_metadata['year']}"
    
    # Get filing URL and metadata
    filing = filings["filings"][0]
    filing_url = filing["linkToFilingDetails"]
    
    # Process key sections (Risk Factors and MD&A)
    result = api.process_filing_for_rag(filing_url, ["1A", "7"])
    
    if "error" in result:
        return f"Error processing filing: {result['error']}"
        
    # Search for relevant chunks
    results = api.text_store.search(query, k=3)
    
    if not results:
        return "No relevant information found in the filing."
    
    # Format response with context
    response = []
    for r in results:
        if r.get("context"):
            response.append(f"\nContext: {r['context']}")
        response.append(f"Content: {r['text'][:500]}...")
        if r.get("relevance_score"):
            response.append(f"Relevance: {r['relevance_score']:.2f}\n")
            
    return "\n".join(response)