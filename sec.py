#!/usr/bin/env python3
"""
SEC Unified Agent - Demonstrating multiple tool usage with SEC-API

This script creates a unified agent that combines multiple SEC-API tools:
1. Filing Search: Find SEC filings using various criteria
2. Section Extraction: Extract specific sections from filings
3. XBRL Data: Extract structured financial data from filings

Usage:
  python sec.py               # Interactive mode
  python sec.py "your query"  # Single query mode
"""

from tools.query_api import sec_filing_search
from tools.section_api import sec_section_extractor
from tools.xbrl_api import sec_xbrl_extractor
from tools.rag_tool import sec_rag_processor
from tools.logger import logger, log_user_interaction
from langchain.tools import StructuredTool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
import os
import sys
from dotenv import load_dotenv
import dateparser
from datetime import datetime, timedelta
import re
from langchain.callbacks.base import BaseCallbackHandler
import json

# Load environment variables
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Global LLM instance for RAG
agent_llm = None


def resolve_relative_date_terms(query):
    """Uses dateparser to handle relative date terms in SEC queries"""
    # Define patterns for common date expressions
    date_patterns = [
        r"(latest|most\s+recent|current)\s+(10-[KQ]|annual|quarterly|report|filing)",  # noqa: E501
        r"(last|this)\s+(year|quarter|month)",
        r"\b(recent|latest)\b",
    ]

    modified_query = query
    date_context = {}

    for pattern in date_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            date_text = match.group(0)
            # Use dateparser to understand the relative date
            parsed_date = dateparser.parse(
                date_text, settings={"RELATIVE_BASE": datetime.now()}
            )

            if parsed_date:
                # Format for SEC API
                if any(
                    word in date_text.lower() for word in ["year", "annual"]
                ):
                    year = parsed_date.year
                    date_range = f"filedAt:[{year}-01-01 TO {year}-12-31]"
                    date_context["date_info"] = f"from {year}"
                elif any(
                    word in date_text.lower()
                    for word in ["quarter", "quarterly"]
                ):
                    # For quarters, use a 3-month range
                    end_date = parsed_date.strftime("%Y-%m-%d")
                    start_date = (parsed_date - timedelta(days=90)).strftime(
                        "%Y-%m-%d"
                    )
                    date_range = f"filedAt:[{start_date} TO {end_date}]"
                    # Use noqa E501 to ignore line length for this specific formatted string
                    date_context["date_info"] = (
                        f"from {start_date} "  # noqa: E501
                        f"to {end_date}"
                    )
                else:
                    # Default to 180 days for "recent", "latest", etc.
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (
                        datetime.now() - timedelta(days=180)
                    ).strftime("%Y-%m-%d")
                    date_range = f"filedAt:[{start_date} TO {end_date}]"
                    # Use noqa E501 to ignore line length for this specific formatted string
                    date_context["date_info"] = (
                        f"from {start_date} "  # noqa: E501
                        f"to {end_date}"
                    )

                date_context["original_term"] = date_text
                modified_query = modified_query.replace(date_text, date_range)
                logger.info(
                    f"Date resolver transformed: '{date_text}' => '{date_range}'"  # noqa: E501
                )

    return modified_query, date_context


class RAGToolHandler(BaseCallbackHandler):
    """Callback handler that processes tool outputs with RAG."""
    
    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        """Process outputs with RAG if they have RAG metadata."""
        tool_name = kwargs.get("name", "")
        
        # Only process outputs from the section extractor
        if tool_name != "sec_section_extractor":
            return
        
        # Check if the output is a dictionary with RAG metadata
        if not isinstance(output, dict) or "metadata" not in output:
            return
        
        metadata = output.get("metadata", {})
        if not metadata.get("needs_rag_processing", False):
            return
        
        # Get the temporary file path and the original query
        temp_file_path = metadata.get("temp_file_path", "")
        input_args = kwargs.get("inputs", {})
        if isinstance(input_args, str):
            # Try to parse JSON if it's a string
            try:
                input_args = json.loads(input_args)
            except:
                input_args = {}
                
        original_query = ""
        if isinstance(input_args, dict):
            original_query = input_args.get("original_query", "")
        
        # Process the section content with RAG
        if os.path.exists(temp_file_path) and original_query:
            try:
                logger.info(f"Processing section content with RAG: {temp_file_path}")
                processed_result = sec_rag_processor(
                    query=original_query, 
                    temp_file_path=temp_file_path
                )
                if processed_result:
                    # Replace the raw text with the processed result
                    output["raw_text"] = processed_result
                    logger.info("RAG processing successful")
                else:
                    logger.warning("RAG processing returned empty result")
            except Exception as e:
                logger.error(f"Error in RAG processing: {str(e)}")


def setup_agent():
    """Set up and return a langchain agent with SEC tools"""
    logger.info("Setting up SEC agent with tools")

    # Create the LLM
    llm = ChatOpenAI(temperature=0)
    
    # Store the LLM instance globally for RAG use
    global agent_llm
    agent_llm = llm

    # Create tools using our standardized functions
    search_tool = StructuredTool.from_function(
        func=sec_filing_search,
        name="sec_filing_search",
        description=(
            "Search for SEC filings. Use Elasticsearch query syntax: "
            '\'ticker:MSFT AND formType:"10-K" AND '
            "filedAt:[2023-01-01 TO 2023-12-31]'. "  # noqa: E501
            "Important: For dictionary input, you must include a 'query' key "
            "with the query string."
        ),
    )

    section_tool = StructuredTool.from_function(
        func=sec_section_extractor,
        name="sec_section_extractor",
        description=(
            "Extract a specific section from an SEC filing. For large sections, you MAY "
            "need to process the result with sec_rag_processor to extract relevant information."
        ),  # noqa: E501
    )

    xbrl_tool = StructuredTool.from_function(
        func=sec_xbrl_extractor,
        name="sec_xbrl_extractor",
        description=(
            "Extract financial data from SEC filings. For detailed results, you MAY "
            "need to process the result with sec_rag_processor to extract relevant information."
        ),  # noqa: E501
    )
    
    rag_tool = StructuredTool.from_function(
        func=sec_rag_processor,
        name="sec_rag_processor",
        description=(
            "Process large text content with Retrieval-Augmented Generation (RAG). "
            "Use this when a section's content is too large to process directly. "
            "Pass the text, your query, and optional source information."
        ),
    )

    # Create agent with all tools
    agent = initialize_agent(
        [search_tool, section_tool, xbrl_tool, rag_tool],
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        # Add the system message as part of the agent's configuration
        agent_kwargs={
            "system_message": """You are an SEC data specialist that ONLY uses data from the SEC-API.
If information cannot be found through the SEC-API tools provided to you, respond with "There is no SEC data available for that query."

IMPORTANT QUERY FORMAT:
- When searching for SEC filings, use string queries like: "ticker:MSFT AND formType:\"10-K\" AND filedAt:[2023-01-01 TO 2023-12-31]"
- For specific sections, you MUST use the correct section code:
  * "1A" for Risk Factors
  * "7" for Management's Discussion and Analysis (MD&A)
  * "1" for Business
  * Other valid 10-K section codes: 1B, 1C, 2, 3, 4, 5, 6, 7A, 8, 9, 9A, 9B, 9C, 10, 11, 12, 13, 14, 15
- Do NOT use descriptive names like "risk factors" - only use the numeric/alphanumeric codes

PROCESSING LARGE CONTENT:
- When a section is too large, follow these steps:
  1. First extract the section using sec_section_extractor
  2. Check if the result metadata contains "needs_rag_processing": true and "temp_file_path" 
  3. If a temp file path is available, use sec_rag_processor with:
     - file_path: The path provided in the section extractor response metadata
     - query: The original user question
     - source: Descriptive source information (e.g., "section 1A of Apple's 2023 10-K")
  4. If no temp file path is available but the content is large, you can still use sec_rag_processor with:
     - text: The truncated text from the section extractor
     - query: The original user question
     - source: Descriptive source information

CRITICAL - DATE INFORMATION REQUIREMENT:
- Always include the specific date information at the beginning of your response
- When relative terms like "latest" or "most recent" are used, begin your response with: "Based on [company]'s [filing type] from [specific date/period]..."
- This helps users understand exactly which filing you're referencing
- NEVER omit this date information when relative time terms were used in the query

When presenting financial metrics:
1. Always include the breakdown components along with totals
2. Format numerical data consistently (e.g., "$X million" or "$XB")
3. Include relevant time periods with each figure
4. Present hierarchical relationships between totals and components"""
        },
    )

    return agent


def process_query(question):
    """Processes a single query using the SEC agent."""
    print(f"\nProcessing: {question}\n")

    # Log the user query
    log_user_interaction(question)

    # Simple confirmation - just echo back the query
    print(f"\nI'll search for information about \"{question}\". Is that correct?")
    confirmation = input("(yes/y/no/n): ").lower()
    
    if confirmation not in ["yes", "y"]:
        print("Query canceled. Please try again with a clearer question.")
        return

    # Pre-process question for relative date terms
    modified_question, date_context = resolve_relative_date_terms(question)
    if modified_question != question:
        logger.info(
            f"Modified question with date resolution: {modified_question}"
        )

    # Initialize a fresh agent
    logger.info("Creating fresh agent instance")
    
    # Create the base agent
    agent = setup_agent()
    
    # Add the RAG handler to all agent invocations
    rag_handler = RAGToolHandler()
    
    # Process the question
    logger.info(f"Agent processing question: {modified_question}")

    # Add date context information to the input if available
    input_with_context = modified_question
    if date_context and "date_info" in date_context:
        orig_term = date_context.get("original_term", "relative date term")
        # Break long f-string for readability and linting
        input_with_context = (
            f"{modified_question}\n\nCRITICAL INSTRUCTION: The query contained "
            f"the term '{orig_term}', which refers to documents "
            f"{date_context['date_info']}. You MUST begin your response with "
            f"'Based on [company]'s [filing type] "
            f"{date_context['date_info']}...' to clearly indicate the time "
            f"period that was searched."
        )  # noqa: E501
    
    # Process the query with the RAG handler
    response = agent.invoke(
        {"input": input_with_context},
        callbacks=[rag_handler]
    )
    
    logger.info("Agent completed processing")

    # Log the agent's response
    log_user_interaction(question, response["output"])

    print("\nANSWER:")
    print(response["output"])


def main():
    """Main function for interactive SEC agent or single query execution"""
    try:
        logger.info("Starting SEC Unified Agent")

        # Check for required API keys
        if not SEC_API_KEY:
            logger.error("SEC_API_KEY environment variable not set")  # noqa: E501
            print("Error: SEC_API_KEY environment variable not set")
            print("Please set this in your .env file")
            sys.exit(1)

        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")  # noqa: E501
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set this in your .env file")
            sys.exit(1)

        # Check if a query is provided via command line argument
        if len(sys.argv) > 1:
            # Join all arguments in case the query has spaces
            query = " ".join(sys.argv[1:])
            process_query(query)
        else:
            # Run interactive mode if no arguments are provided
            print("SEC Unified Agent - Demo with Multiple Tools")
            print("-" * 70)
            print("This agent can:")
            print("1. Search for SEC filings")
            print(
                "2. Extract specific sections from filings (with smart processing)"
            )
            print(
                "3. Extract XBRL financial data from filings"
            )
            print("-" * 70)

            # Example questions to demonstrate multi-tool usage
            demo_questions = [
                "What were the risk factors mentioned in Tesla's most recent 10-K filing?",
                "What was Apple's total revenue in their latest 10-Q of 2023?",
                # Use noqa E501 to ignore line length for this specific long string
                (
                    "Find Microsoft's 10-K filings from 2023 and extract the "  # noqa E501
                    "Management Discussion section"
                ),
            ]

            print("Demo Questions:")
            for i, question in enumerate(demo_questions, 1):
                print(f"{i}. {question}")

            # Interactive question answering loop
            while True:
                print("\n" + "-" * 70)
                # Break long input prompt for readability
                choice = input(
                    "Enter question number, your own question, or 'quit' to exit: "  # noqa: E501
                )

                if choice.lower() in ["quit", "exit", "q"]:
                    logger.info("User requested exit")
                    break

                # Get question (either from demo or custom)
                question = ""
                if choice.isdigit() and 1 <= int(choice) <= len(
                    demo_questions
                ):
                    question = demo_questions[int(choice) - 1]
                else:
                    question = choice

                process_query(question)  # Use the refactored function

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
# Add newline at the end
