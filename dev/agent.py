from tools.query_api import sec_filing_search
from tools.xbrl_api import sec_xbrl_extractor
from langchain.tools import StructuredTool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create the tools
search_tool = StructuredTool.from_function(
    func=sec_filing_search,
    name="sec_filing_search",
    description=sec_filing_search.__doc__
)

xbrl_tool = StructuredTool.from_function(
    func=sec_xbrl_extractor,
    name="sec_xbrl_extractor",
    description=sec_xbrl_extractor.__doc__
)

# Create a simple agent with both tools
llm = ChatOpenAI(temperature=0)
agent = initialize_agent([search_tool, xbrl_tool], llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# Test questions
test_questions = [
    # "Find Tesla's 10-K filings from 2023",
    "What was Tesla's total revenue in their most recent 10-K?"
]

# Run tests
for question in test_questions:
    print(f"\n\nTESTING: {question}")
    response = agent.invoke({"input": question})
    print("\nAGENT RESPONSE:")
    print(response["output"])
    
    # Pause between tests
    input("\nPress Enter to continue to the next test...")


# from tools.query_api import sec_filing_search
# from tools.xbrl_api import sec_xbrl_extractor
# from langchain.tools import StructuredTool
# from langchain.agents import AgentType, initialize_agent
# from langchain_openai import ChatOpenAI
# import os

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()
# SEC_API_KEY = os.getenv("SEC_API_KEY")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Add this to .env

# # Define the second tool (XBRL extractor)
# def sec_xbrl_extractor(htm_url=None, accession_number=None, return_format="json"):
#     """
#     SEC EDGAR XBRL-to-JSON Converter API - Complete Functionality Guide

#     This API converts XBRL data from SEC filings (10-K, 10-Q) into structured JSON format, 
#     providing access to financial statements and tagged data since 2009.

#     Args:
#         htm_url (str, optional): Full URL to the filing's HTML
#         accession_number (str, optional): SEC accession number
#         return_format (str, optional): Output format, either "json" (default) or "raw"

#     Returns:
#         dict: XBRL data in JSON or raw format, or error details
#     """
#     from sec_api import XbrlApi
#     try:
#         if not htm_url and not accession_number:
#             return {"error": "Must provide either htm_url or accession_number", "type": "ValueError"}
#         xbrl_api = XbrlApi(api_key=SEC_API_KEY)
#         if htm_url:
#             if return_format == "raw":
#                 raw_data = xbrl_api.get_raw_xbrl(htm_url=htm_url)
#                 return {"xbrl_raw": raw_data, "filing_url": htm_url}
#             else:
#                 return xbrl_api.xbrl_to_json(htm_url=htm_url)
#         else:
#             if return_format == "raw":
#                 raw_data = xbrl_api.get_raw_xbrl(accession_no=accession_number)
#                 return {"xbrl_raw": raw_data, "filing_url": f"Fetched by accession {accession_number}"}
#             else:
#                 return xbrl_api.xbrl_to_json(accession_no=accession_number)
#     except Exception as e:
#         return {"error": str(e), "type": str(type(e).__name__)}

# # Create tools
# search_tool = StructuredTool.from_function(
#     func=sec_filing_search,
#     name="sec_filing_search",
#     description=sec_filing_search.__doc__
# )

# xbrl_tool = StructuredTool.from_function(
#     func=sec_xbrl_extractor,
#     name="sec_xbrl_extractor",
#     description=sec_xbrl_extractor.__doc__
# )

# # Create agent with both tools
# llm = ChatOpenAI(temperature=0)
# agent = initialize_agent(
#     [search_tool, xbrl_tool],
#     llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     verbose=True
# )

# # Test questions
# test_questions = [
#     "Find Tesla's latest 10-K and extract its total assets.",
#     "Get Apple's 2023 10-Q filings and show their revenue.",
#     "Find the most recent 10-K for Microsoft and check its cash flow from operations."
# ]

# # Run tests
# for question in test_questions:
#     print(f"\n\nTESTING: {question}")
#     response = agent.invoke({"input": question})
#     print("\nAGENT RESPONSE:")
#     print(response["output"])
#     input("\nPress Enter to continue...")