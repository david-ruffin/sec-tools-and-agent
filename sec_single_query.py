#!/usr/bin/env python3
"""
SEC Unified Agent - Demonstrating multiple tool usage with SEC-API

This script creates a unified agent that combines multiple SEC-API tools:
1. Filing Search: Find SEC filings using various criteria
2. Section Extraction: Extract specific sections from filings  
3. XBRL Data: Extract structured financial data from filings

Usage: python sec_unified_agent.py
"""

from tools.query_api import sec_filing_search
from tools.section_api import sec_section_extractor
from tools.xbrl_api import sec_xbrl_extractor
from tools.logger import logger, log_user_interaction
from langchain.tools import StructuredTool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def setup_agent():
    """Set up and return a langchain agent with SEC tools"""
    logger.info("Setting up SEC agent with tools")
    
    # Create tools using our standardized functions
    search_tool = StructuredTool.from_function(
        func=sec_filing_search,
        name="sec_filing_search",
        description="Search for SEC filings. Use Elasticsearch query syntax: 'ticker:MSFT AND formType:\"10-K\" AND filedAt:[2023-01-01 TO 2023-12-31]'. Important: For dictionary input, you must include a 'query' key with the query string."
    )

    section_tool = StructuredTool.from_function(
        func=sec_section_extractor,
        name="sec_section_extractor",
        description="Extract a specific section from an SEC filing. For large sections, use mode='outline' first, then mode='summary' for key details."
    )
    
    xbrl_tool = StructuredTool.from_function(
        func=sec_xbrl_extractor,
        name="sec_xbrl_extractor",
        description="Extract financial data from SEC filings. To avoid exceeding token limits: 1) First call without statement_type to see available statements, 2) Then request a specific statement using the exact name from the API response (like statement_type='StatementsOfIncome'), 3) You can further filter by specific metrics like metrics=['Revenue', 'NetIncome']."
    )

    # Create the LLM
    llm = ChatOpenAI(temperature=0)
    
    # Create agent with all tools
    agent = initialize_agent(
        [search_tool, section_tool, xbrl_tool],
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
- For specific sections, common codes are: "1A" (Risk Factors), "7" (MD&A), "1" (Business)

For XBRL financial data:
1. First get available statements (call without statement_type)
2. Then request a specific statement using the EXACT statement name returned in the API response (common names include "StatementsOfIncome", "BalanceSheets", "StatementsOfCashFlows")
3. Optionally filter for specific metrics (e.g., metrics=["Revenue", "NetIncome"])

For large sections of text:
1. First use mode='outline' to get the structure
2. Then use mode='summary' to get key details
3. Finally, if needed, request specific portions

When presenting financial metrics:
1. Always include the breakdown components along with totals
2. Format numerical data consistently (e.g., "$X million" or "$XB")
3. Include relevant time periods with each figure
4. Present hierarchical relationships between totals and components

EXAMPLE FORMATS:
For revenue: "Total revenue: $81.8B for Q3 2023 (Product: $60.6B, Services: $21.2B)"
For income: "Net income: $19.9B for Q3 2023 (Operating income: $23.1B, Tax expense: $3.2B)"

This approach helps manage context length for complex SEC filings."""
        }
    )
    
    return agent

def main():
    """Main function for interactive SEC agent"""
    try:
        logger.info("Starting SEC Unified Agent")
        print("SEC Unified Agent - Demo with Multiple Tools")
        print("-" * 70)
        print("This agent can:")
        print("1. Search for SEC filings")
        print("2. Extract specific sections from filings (with smart chunking)")
        print("3. Extract XBRL financial data from filings (with statement filtering)")
        print("-" * 70)
        
        # Check for required API keys
        if not SEC_API_KEY:
            logger.error("SEC_API_KEY environment variable not set")
            print("Error: SEC_API_KEY environment variable not set")
            print("Please set this in your .env file")
            sys.exit(1)
            
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set this in your .env file")
            sys.exit(1)
        
        # Example questions to demonstrate multi-tool usage
        demo_questions = [
            "What were the risk factors mentioned in Tesla's most recent 10-K filing?",
            "What was Apple's total revenue in their latest 10-Q of 2023?",
            "Find Microsoft's 10-K filings from 2023 and extract the Management Discussion section"
        ]
        
        print("Demo Questions:")
        for i, question in enumerate(demo_questions, 1):
            print(f"{i}. {question}")
        
        # Interactive question answering loop
        while True:
            print("\n" + "-" * 70)
            choice = input("Enter question number, your own question, or 'quit' to exit: ")
            
            if choice.lower() in ["quit", "exit", "q"]:
                logger.info("User requested exit")
                break
                
            # Get question (either from demo or custom)
            question = ""
            if choice.isdigit() and 1 <= int(choice) <= len(demo_questions):
                question = demo_questions[int(choice) - 1]
            else:
                question = choice
                
            print(f"\nProcessing: {question}\n")
            
            # Log the user query
            log_user_interaction(question)
            
            # Initialize a fresh agent for each question to reset context
            logger.info("Creating fresh agent instance to reset context")
            agent = setup_agent()
            
            # Process the question
            logger.info(f"Agent processing question: {question}")
            response = agent.invoke({"input": question})
            logger.info(f"Agent completed processing")
            
            # Log the agent's response
            log_user_interaction(question, response["output"])
            
            print("\nANSWER:")
            print(response["output"])
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 