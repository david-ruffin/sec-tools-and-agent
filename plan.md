# SEC Tools RAG Implementation Plan

## Overview

This plan outlines the implementation of Retrieval-Augmented Generation (RAG) for all SEC-API tools in the project. The goal is to ensure that all text responses from SEC-API tools are processed through a RAG pipeline, allowing the agent to answer questions about even very large documents without context window limitations.

## Implementation Steps

### 1. Create RAG Utility Module (`tools/rag_handler.py`)

**Purpose:** Create a reusable utility to handle the RAG processing for any tool output.

**Code Implementation:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any, Dict, Optional

def rag_process(text_content: str, user_query: str, llm: Any) -> str:
    """
    Process text content using RAG approach.
    
    Args:
        text_content: The full text to process
        user_query: The original user question
        llm: LLM instance to use for answering
        
    Returns:
        A string containing the RAG-processed answer
    """
    # Skip RAG for empty or very short content
    if not text_content or len(text_content) < 500:
        return text_content
        
    # 1. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text_content)
    
    # 2. Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # 3. Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 4. Create and run QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # 5. Process query and return result
    system_prompt = "You are analyzing SEC filing text to answer a user's question. Based on the relevant sections retrieved, provide a detailed and accurate response that directly addresses the user's question."
    
    enhanced_query = f"Using only the provided context, answer this question: {user_query}"
    result = qa_chain({"query": enhanced_query})
    return result["result"]
```

### 2. Update Tool Wrappers

#### 2.1 Modify `tools/section_api.py`

**Changes:**
- Remove the `mode` parameter logic that truncates content
- Always return the full section text
- Add `original_query` parameter

```python
@log_api_call
def sec_section_extractor(filing_url, section_code, return_type="text", original_query=None):
    """
    Extract a specific section from an SEC filing.
    
    Args:
        filing_url: URL to the SEC filing HTML document
        section_code: Section identifier (e.g., '1A' for Risk Factors, '7' for MD&A)
        return_type: Format of extraction ('text' or 'html')
        original_query: The original user query
    
    Returns:
        Dict with section data, metadata and status information
    """
    try:
        # Validate inputs
        if not filing_url:
            return {
                "status": "error",
                "message": "Filing URL is required",
                "data": None
            }
        
        if not section_code:
            return {
                "status": "error",
                "message": "Section code is required",
                "data": None
            }
        
        # Initialize the Extractor API
        extractor_api = ExtractorApi(api_key=SEC_API_KEY)
        
        # Extract the section
        full_content = extractor_api.get_section(filing_url, section_code, return_type)
        
        # Validate the response
        if not full_content or (isinstance(full_content, str) and len(full_content) < 50):
            return {
                "status": "no_results",
                "message": f"Section {section_code} not found or contains minimal content",
                "filing_url": filing_url,
                "section_code": section_code,
                "data": None
            }
        
        # Get content length
        content_length = len(full_content)
        
        # Create metadata
        metadata = {
            "total_length": content_length,
            "section_code": section_code,
            "filing_url": filing_url
        }
        
        # Return full content always
        return {
            "status": "success",
            "message": f"Successfully extracted section {section_code}",
            "data": full_content,
            "metadata": metadata,
            "raw_text": full_content  # Store full text for RAG processing
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": str(type(e).__name__),
            "filing_url": filing_url if 'filing_url' in locals() else None,
            "section_code": section_code if 'section_code' in locals() else None,
            "data": None
        }
```

#### 2.2 Update `tools/xbrl_api.py`

Similar updates to ensure all text is processed through RAG.

#### 2.3 Update `tools/query_api.py`

Similar updates to ensure all text is processed through RAG.

### 3. Modify Main Agent (`sec.py`)

#### 3.1 Update Imports and Global Variables

```python
from tools.rag_handler import rag_process
import functools

# Global LLM instance for RAG
agent_llm = None
```

#### 3.2 Modify `setup_agent` Function

```python
def setup_agent():
    """Set up and return a langchain agent with SEC tools"""
    logger.info("Setting up SEC agent with tools")

    # Create the LLM
    llm = ChatOpenAI(temperature=0)
    
    # Store the LLM instance globally for RAG use
    global agent_llm
    agent_llm = llm
    
    # Create tools using our standardized functions and wrap with RAG processing
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
            "Extract a specific section from an SEC filing. Returns the full section "
            "content, which will be automatically processed to answer your question."
        ),  # noqa: E501
    )

    xbrl_tool = StructuredTool.from_function(
        func=sec_xbrl_extractor,
        name="sec_xbrl_extractor",
        description=(
            "Extract financial data from SEC filings. The full response will be "
            "automatically processed to answer your question about the data."
        ),  # noqa: E501
    )

    # Create agent with all tools
    agent = initialize_agent(
        [search_tool, section_tool, xbrl_tool],
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        # Update the system message
        agent_kwargs={
            "system_message": """You are an SEC data specialist that ONLY uses data from the SEC-API.
If information cannot be found through the SEC-API tools provided to you, respond with "There is no SEC data available for that query."

IMPORTANT: All responses from SEC-API tools are automatically processed using Retrieval-Augmented Generation (RAG) to extract the most relevant information for the user's query.

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
```

#### 3.3 Add RAG processing to `process_query` Function

```python
def process_query(question):
    """Processes a single query using the SEC agent."""
    print(f"\nProcessing: {question}\n")

    # Log the user query
    log_user_interaction(question)

    # Pre-process question for relative date terms
    modified_question, date_context = resolve_relative_date_terms(question)
    if modified_question != question:
        logger.info(
            f"Modified question with date resolution: {modified_question}"
        )

    # Initialize a fresh agent
    logger.info("Creating fresh agent instance")
    agent = setup_agent()

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

    # Custom callback handler to intercept and process tool results
    class RAGToolHandler(BaseCallbackHandler):
        def on_tool_end(self, output, **kwargs):
            # Skip if output isn't a dictionary or doesn't have raw_text
            if not isinstance(output, dict) or "raw_text" not in output:
                return output
                
            # Skip if raw_text is not a string or is very short
            if not isinstance(output["raw_text"], str) or len(output["raw_text"]) < 500:
                return output
                
            # Process with RAG
            logger.info("Processing large text response with RAG")
            rag_result = rag_process(output["raw_text"], question, agent_llm)
            
            # Add the RAG result to the output
            output["data"] = rag_result
            output["message"] = "Response processed with RAG to extract relevant information"
            
            return output

    # Create an instance of the handler
    rag_handler = RAGToolHandler()
    
    # Add the handler to the agent's callbacks
    response = agent.invoke(
        {"input": input_with_context},
        callbacks=[rag_handler]
    )
    
    logger.info("Agent completed processing")

    # Log the agent's response
    log_user_interaction(question, response["output"])

    print("\nANSWER:")
    print(response["output"])
```

### 4. Update Requirements

Update `requirements.txt` to include:

```
sec-api>=1.0.0
langchain>=0.0.267
langchain-community>=0.0.8
langchain-huggingface>=0.0.1
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
```

### 5. Testing Plan

1. **Basic Functionality Test:**
   - Ensure all SEC-API tools return results correctly
   - Verify RAG processing is applied to all text responses

2. **Large Content Test:**
   - Test with Apple's Risk Factors section (known to be large)
   - Verify the agent can answer specific questions about content deep in the section

3. **Integration Test:**
   - Test chaining multiple tools together in a single query
   - Verify the agent correctly uses the RAG-processed information

## Timeline

1. **Phase 1 (Day 1):**
   - Create `rag_handler.py`
   - Update `requirements.txt`
   - Modify `section_api.py`

2. **Phase 2 (Day 1-2):**
   - Modify `query_api.py` and `xbrl_api.py`
   - Update `sec.py` with RAG integration

3. **Phase 3 (Day 2):**
   - Testing and debugging
   - Documentation updates

## Success Criteria

- Agent can process and answer questions about very large sections (like Risk Factors)
- No context window errors occur with large documents
- Agent provides accurate, detailed answers that correctly reference content from any part of large documents
- Performance remains within acceptable limits (processing time < 15 seconds) 