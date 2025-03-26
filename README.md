# SEC-FLOYD: SEC Filings LLM-Oriented YAML-Driven Agent

## Project Overview

SEC-FLOYD is an LLM-powered agent system designed to extract, analyze, and present financial data from SEC filings. The project creates a unified interface for querying multiple aspects of SEC filings through natural language, leveraging the sec-api-python library and LLM agents.

## Goals

1. **Unified Financial Data Access**: Create a single conversational interface to SEC filings data
2. **Context Management**: Handle large financial documents efficiently through progressive disclosure
3. **Structured Data Extraction**: Extract specific financial metrics from XBRL data
4. **Multiple Search Capabilities**: Enable search by company, filing type, date range, and other criteria
5. **Section-Specific Analysis**: Extract and analyze specific sections of interest from filings
6. **Comprehensive SEC-API Coverage**: Implement all APIs offered by the sec-api-python library

## Current Approach

The project combines three key components:

1. **SEC-API Integration**: We use the sec-api-python library to access:
   - Filing search capabilities (finding specific documents)
   - Section extraction (pulling specific sections from filings)
   - XBRL data extraction (accessing structured financial metrics)

2. **LLM Agent Framework**: We use LangChain to create a function-calling agent that:
   - Interprets natural language queries
   - Selects appropriate SEC-API tools
   - Manages complex multi-step workflows (search → extract → filter)
   - Presents results in a concise, readable format

3. **Progressive Data Retrieval**: To handle large documents:
   - First retrieve document metadata or outlines
   - Then target specific sections or statements
   - Finally extract specific metrics or paragraphs of interest

## Recommended Enhancements

To improve the accuracy and reliability of financial data extraction, the following enhancements are recommended:

1. **Enhanced XBRL Data Validation**:
   - Add data validation logic to clearly identify and label aggregate values vs. segment breakdowns
   - Prevent confusion between product revenue and total revenue by properly distinguishing hierarchical relationships
   - Implement verification checks to ensure consistency between different views of the same data

2. **Financial Data Interpretation Guidelines**:
   - Modify the system prompt to include specific guidelines for interpreting financial data hierarchies
   - Add instructions for handling common metrics like revenue that appear with multiple segment breakdowns
   - Create explicit rules to prioritize non-segmented values when reporting aggregate metrics like "total revenue"

3. **Verification Steps**:
   - Implement quick verification processes to confirm that values match across different views of the same data
   - Add checks to ensure that total revenue = sum of product and service revenue
   - Build in safeguards against misinterpretation of segmented financial data

These changes would ensure the agent not only retrieves accurate data but also interprets it correctly when answering financial questions.

## Current Status

- ✅ Basic agent framework implemented with SEC-API integration
- ✅ Three core tools integrated (search, section extraction, XBRL extraction)
- ✅ Fixed XBRL tool to correctly use standardized statement names
- ✅ Implemented filtering capabilities for financial metrics 
- ✅ Context management implemented through progressive disclosure approaches
- ✅ Interactive CLI interface for testing and demonstration
- 🔄 Currently testing with real-world financial queries
- ⏱️ Future work: Web interface, visualization, historical comparisons

## Technical Components

### Main Scripts

- `sec_unified_agent.py`: The main entry point that sets up the agent with all tools and handles user interaction. Uses LangChain for agent orchestration.

### Core Tools (Currently Implemented)

#### 1. Filing Search Tool (`query_api.py`)
- **Function**: `sec_filing_search`
- **Purpose**: Find SEC filings based on various search criteria
- **Features**:
  - Elasticsearch-compatible query syntax
  - Search by ticker, form type, date ranges, etc.
  - Returns filing metadata and links
- **Example Query**: `ticker:AAPL AND formType:"10-Q" AND filedAt:[2022-01-01 TO 2023-01-01]`
- **Implementation Details**:
  - Uses `QueryApi` class from sec-api-python
  - Handles both string and dictionary query formats
  - Processes Elasticsearch-format query strings
  - Returns standardized response with metadata
  - API Call Structure:
    ```python
    query_api = QueryApi(api_key=SEC_API_KEY)
    query_result = query_api.get_filings(query_or_params)
    ```
  - Error handling for network issues and invalid queries
  - Response normalization for consistent agent interaction

#### 2. Section Extraction Tool (`section_api.py`)
- **Function**: `sec_section_extractor`
- **Purpose**: Extract specific sections from SEC filings
- **Features**:
  - Extract by section name or number
  - Smart chunking for large sections
  - Multiple modes: full text, summary, outline
- **Common Sections**: "1A" (Risk Factors), "7" (MD&A), "1" (Business)
- **Implementation Details**:
  - Uses `ExtractorApi` class from sec-api-python
  - Supports accession number or URL-based lookups
  - Multiple extraction modes:
    - `full`: Returns complete section text
    - `summary`: Returns condensed section overview
    - `outline`: Returns section structure only
  - API Call Structure:
    ```python
    extractor_api = ExtractorApi(api_key=SEC_API_KEY)
    if section:
        section_text = extractor_api.get_section(
            htm_url=htm_url or None,
            accession_no=accession_number or None,
            section=section,
            return_type=mode or "full"
        )
    else:
        section_text = extractor_api.get_all_sections(
            htm_url=htm_url or None,
            accession_no=accession_number or None
        )
    ```
  - Handles empty sections, missing documents
  - Returns standardized response with section data
  - Content management with mode-based retrieval

#### 3. XBRL Data Extraction Tool (`xbrl_api.py`)
- **Function**: `sec_xbrl_extractor`
- **Purpose**: Extract structured financial data from XBRL filings
- **Features**:
  - Return all available financial statements
  - Filter by specific statement type (using exact statement names from API)
  - Further filter by specific financial metrics
  - Handle standardized statement naming conventions
- **Common Statement Names**:
  - "StatementsOfIncome" (income statements)
  - "BalanceSheets" (balance sheets)
  - "StatementsOfCashFlows" (cash flow statements)
- **Implementation Details**:
  - Uses `XbrlApi` class from sec-api-python
  - Supports both raw XBRL and JSON format outputs
  - Uses three-step approach for context management:
    1. Fetch available statements when no statement_type provided
    2. Fetch complete statement when statement_type provided
    3. Filter by specific metrics when metrics parameter provided
  - API Call Structure for JSON format:
    ```python
    xbrl_api = XbrlApi(api_key=SEC_API_KEY)
    if htm_url:
        full_data = xbrl_api.xbrl_to_json(htm_url=htm_url)
    else:
        full_data = xbrl_api.xbrl_to_json(accession_no=accession_number)
    ```
  - API Call Structure for raw format:
    ```python
    xbrl_api = XbrlApi(api_key=SEC_API_KEY)
    if htm_url:
        raw_data = xbrl_api.get_raw_xbrl(htm_url=htm_url)
    else:
        raw_data = xbrl_api.get_raw_xbrl(accession_no=accession_number)
    ```
  - Statement type handling relies on exact statement names from SEC-API
  - Metrics filtering uses case-insensitive partial matching
  - Returns standardized response with financial data
  - Progressive disclosure to manage token context limits

### Agent Implementation (`sec_unified_agent.py`)
- **Framework**: LangChain
- **LLM**: OpenAI API (configurable)
- **Agent Type**: Function-calling agent (OPENAI_FUNCTIONS)
- **Implementation Details**:
  - Tools wrapped as LangChain StructuredTool objects
  - System message provides detailed context and instructions
  - Error handling for parsing and API errors
  - Tools registered with function signatures and descriptions
  - Detailed system message guides multi-step workflows
  - Interactive CLI for demonstration

## Complete List of SEC-API-Python APIs

The sec-api-python library offers the following APIs, with the first three already implemented in our project:

### EDGAR Filing Search & Download APIs
1. ✅ **SEC Filing Search and Full-Text Search API** - Find filings across the entire EDGAR database
2. **Real-Time Filing Stream API** - Receive new filings as they're published
3. **Download API** - Download any SEC filing, exhibit and attached file
4. **PDF Generator API** - Download SEC filings and exhibits as PDF

### Converter & Extractor APIs
5. ✅ **XBRL-to-JSON Converter API + Financial Statements** - Convert XBRL to structured financial data
6. ✅ **10-K/10-Q/8-K Section Extraction API** - Extract specific sections from filings

### Investment Advisers
7. **Form ADV API** - Investment Advisors (Firm & Individual Advisors, Brochures, Schedules)

### Ownership Data APIs
8. **Form 3/4/5 API** - Insider Trading Disclosures
9. **Form 13F API** - Institutional Investment Manager Holdings & Cover Pages
10. **Form 13D/13G API** - Activist and Passive Investor Holdings
11. **Form N-PORT API** - Mutual Funds, ETFs and Closed-End Fund Holdings

### Proxy Voting Records
12. **Form N-PX Proxy Voting Records API** - Fund voting records

### Security Offerings APIs
13. **Form S-1/424B4 API** - Registration Statements and Prospectuses (IPOs, Debt/Warrants Offerings)
14. **Form C API** - Crowdfunding Offerings & Campaigns
15. **Form D API** - Private Security Offerings
16. **Regulation A APIs** - Offering Statements by Small Companies (Form 1-A, Form 1-K, Form 1-Z)

### Structured Material Event Data from Form 8-K
17. **Auditor and Accountant Changes** (Item 4.01)
18. **Financial Restatements & Non-Reliance on Prior Financial Results** (Item 4.02)
19. **Changes of Directors, Board Members and Compensation Plans** (Item 5.02)

### Public Company Data
20. **Directors & Board Members API**
21. **Executive Compensation Data API**
22. **Outstanding Shares & Public Float**
23. **Company Subsidiary API**

### Enforcement Actions, Proceedings, AAERs & SRO Filings
24. **SEC Enforcement Actions**
25. **SEC Litigation Releases**
26. **SEC Administrative Proceedings**
27. **AAER Database API** - Accounting and Auditing Enforcement Releases
28. **SRO Filings Database API** - Self-Regulatory Organization filings

### Other APIs
29. **CUSIP/CIK/Ticker Mapping API**
30. **EDGAR Entities Database API**

## Technical Implementation Pattern

For future API integration, follow this pattern based on the existing implementations:

1. **Create a dedicated module** in the tools directory for each API (e.g., `tools/form_adv_api.py`)
2. **Implement a standardized function** that:
   - Takes logical parameters aligned with the API's capabilities
   - Returns a standardized response format with:
     - `status`: success/error/no_results
     - `message`: Human-readable result summary
     - `data`: The actual data payload
     - Any relevant metadata (e.g., filing_url, accession_number)
   - Includes comprehensive error handling
   - Manages context with progressive disclosure when appropriate
3. **Register the function as a tool** in `sec_unified_agent.py`:
   ```python
   new_tool = StructuredTool.from_function(
       func=new_api_function,
       name="sec_new_function_name",
       description="Clear description of what the tool does and how to use it effectively."
   )
   ```
4. **Update the system message** to explain how to use the new tool
5. **Add appropriate documentation** to this README

## Usage Example

```
SEC Unified Agent - Demo with Multiple Tools
----------------------------------------------------------------------
This agent can:
1. Search for SEC filings
2. Extract specific sections from filings (with smart chunking)
3. Extract XBRL financial data from filings (with statement filtering)
----------------------------------------------------------------------

Demo Questions:
1. What were the risk factors mentioned in Tesla's most recent 10-K filing?
2. What was Apple's total revenue in their latest 10-Q?
3. Find Microsoft's 10-K filings from 2023 and extract the Management Discussion section

----------------------------------------------------------------------
Enter question number, your own question, or 'quit' to exit: 2

Processing: What was Apple's total revenue in their latest 10-Q?

ANSWER:
According to Apple's latest 10-Q filing, their total revenue was $94,840 million for the quarter ended December 30, 2023.
```

## Next Steps and Future Development

1. **Visualization**: Add charts and graphs for financial metrics
2. **Historical Analysis**: Compare metrics across multiple periods
3. **Web Interface**: Create a web UI for easier interaction
4. **Financial Ratio Analysis**: Calculate and explain key financial ratios
5. **Expanded Data Sources**: Add integration with additional financial data sources

## For Developers

To continue development on this project:

1. Set up environment variables in a `.env` file:
   ```
   SEC_API_KEY=your_sec_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

2. Install required packages:
   ```
   pip install langchain langchain-openai sec-api python-dotenv
   ```

3. Run the interactive agent:
   ```
   python sec_unified_agent.py
   ```

4. To extend functionality, follow the existing pattern:
   - Add new tool functions to the appropriate API modules
   - Register new tools in the `setup_agent` function in `sec_unified_agent.py`
   - Update the system message to explain the new capabilities # sec-tools-and-agent
