# SEC Filing Analysis Agent

## 1. Project Description


This started out as a Google AI Studio with "Gemini 2.5 Pro experimental 03-25" with the following prompt "https://drive.google.com/file/d/1EdwU2KFMjvQGPSdgAan4tM0kCsEGi-g2/view?usp=sharing, https://docs.google.com/spreadsheets/d/1MDCj2vW9bYA5w8WCgDYoCkw0N-5-e2POdi3lMyTo2OI/edit?usp=sharing, https://docs.google.com/spreadsheets/d/1UYD14-tjduVwsms4A5JPApYQ1RqpLzqt3oQe5Moj07g/edit?usp=sharing, https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221wR4I8ofLLs0g7aUoj34unkmAufQhVMia%22%5D,%22action%22:%22open%22,%22userId%22:%22107950687044865459625%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing" and it turned into this.


This project implements an AI-powered agent capable of interacting with U.S. Securities and Exchange Commission (SEC) filings using natural language queries. It leverages the `sec-api-python` library to interface with the sec-api.io service and uses Google's Generative AI models (specifically Gemini Flash) via the LangChain framework to understand user queries, orchestrate tool usage, and synthesize answers.

The primary goal is to allow users (potentially financial analysts or researchers familiar with SEC filings) to ask questions about company filings (e.g., 10-K, 10-Q, 8-K) and receive answers based directly on the data retrieved from the SEC EDGAR database via the available tools.

## 2. Core Features

*   **Natural Language Querying:** Users can ask questions in plain English.
*   **SEC Filing Search:** Utilizes `sec_api.QueryApi` to search for specific filings based on criteria like ticker, CIK, form type, filing dates, etc. (`SearchFilingsTool`).
*   **Section Extraction:** Utilizes `sec_api.ExtractorApi` to extract specific, standardized sections from filings (e.g., '1A' - Risk Factors, 'part1item1' - Financial Statements) based on the filing's URL and the correct section identifier (`ExtractSectionTool`).
*   **Retrieval-Augmented Generation (RAG):** When extracting text sections (`return_type='text'`) that exceed a predefined length limit (currently 15,000 characters), the `ExtractSectionTool` automatically:
    *   Chunks the extracted text using `RecursiveCharacterTextSplitter`.
    *   Embeds the chunks using `GoogleGenerativeAIEmbeddings`.
    *   Creates an in-memory vector store using `FAISS`.
    *   Retrieves the most relevant chunks based on the user's original query (passed to the tool).
    *   Returns only the combined relevant chunks, preventing context overflow and focusing the LLM on pertinent information.
*   **Identifier Mapping:** Utilizes `sec_api.MappingApi` to resolve and map between different company identifiers like Ticker, CIK, CUSIP, Name, etc. (`MapIdentifiersTool`).
*   **XBRL Data Extraction:** Utilizes `sec_api.XbrlApi` to parse XBRL data from filings and return key financial statements (Income, Balance Sheet, Cash Flow) in a structured JSON format (`GetXbrlDataTool`).
*   **Agent Framework:** Uses LangChain's `create_tool_calling_agent` and `AgentExecutor` to manage the interaction flow between the LLM, the user query, and the available tools.
*   **Transparent Response Summary:** The agent begins its final answer with a brief summary of the key actions taken (e.g., ticker identified, filing used, section extracted) and includes the source filing URL to enhance user trust and enable quick validation by domain experts.
*   **Command-Line Interface:** Accepts user queries directly as command-line arguments for single-shot execution.
*   **Interactive Mode:** Falls back to an interactive loop if no command-line arguments are provided, allowing multiple questions per session.
*   **Detailed File Logging:** Logs execution steps, tool calls, RAG process details, API responses (counts), errors, and final outputs to timestamped files in a `logs/` directory using Python's `logging` module and `pytz` for PST timestamps.

## 3. Technology Stack

*   **Python:** Core programming language (developed with Python 3.11).
*   **LangChain:** Framework for building LLM applications (`langchain`, `langchain-core`, `langchain-google-genai`, `langchain-text-splitters`, `langchain-community`).
*   **Google Generative AI:** LLM provider (using `gemini-1.5-flash-latest` model).
*   **sec-api-python:** Client library for interacting with the sec-api.io service.
*   **FAISS (CPU):** Library for efficient similarity search, used for the RAG vector store (`faiss-cpu`).
*   **Pydantic:** Used for defining structured input schemas for the LangChain tools.
*   **python-dotenv:** For loading environment variables from a `.env` file.
*   **pytz:** For timezone-aware logging timestamps.

## 4. Setup Instructions

1.  **Prerequisites:**
    *   Python 3.9 or higher installed.
    *   `pip` (Python package installer).
    *   Git (for cloning, if applicable).

2.  **Clone Repository (Optional):** If you haven't already, clone the project repository.
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create Virtual Environment:** It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    ```

4.  **Activate Virtual Environment:**
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    *(You should see `(.venv)` preceding your command prompt.)*

5.  **Install Dependencies:** Install all required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## 5. Configuration

The script requires API keys for both sec-api.io and Google Generative AI. These should be stored securely in an environment file.

1.  **Create `.env` file:** In the root directory of the project, create a file named `.env`.
2.  **Add API Keys:** Add the following lines to the `.env` file, replacing `<YOUR_SEC_API_KEY>` and `<YOUR_GOOGLE_API_KEY>` with your actual keys:
    ```dotenv
    SEC_API_KEY=<YOUR_SEC_API_KEY>
    GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>
    ```
3.  **Security:** Ensure the `.env` file is included in your `.gitignore` file to prevent accidentally committing your API keys.

## 6. Usage

The script can be run in two modes:

1.  **Command-Line Argument Mode:** Provide the query directly as arguments after the script name. Enclose multi-word queries in quotes. The script will execute once for the given query and print the result.

    ```bash
    python sec_agent.py "Your question about SEC filings?"
    ```
    *Example:*
    ```bash
    python sec_agent.py "Are there any new footnote disclosures related to cybersecurity in the latest 10-Q for Tesla?"
    ```

2.  **Interactive Mode:** Run the script without any arguments to enter an interactive loop where you can ask multiple questions.

    ```bash
    python sec_agent.py
    ```
    *Output:*
    ```
    Your Question: <Type your question here and press Enter>
    Assistant:
    <Agent's response>

    Your Question: <Type 'quit' to exit>
    ```

## 7. Code Structure Overview (`sec_agent.py`)

*   **Imports:** Imports necessary libraries (os, json, dotenv, logging, datetime, pytz, sys, pydantic, langchain components, sec_api clients).
*   **API Key Loading:** Loads keys from `.env` using `dotenv`.
*   **Logging Setup:** `setup_logging()` function configures file and console logging with PST timestamps.
*   **SEC API Client Initialization:** Initializes clients (`QueryApi`, `ExtractorApi`, etc.).
*   **Pydantic Schemas:** Defines input structures (`SearchFilingsInput`, `ExtractSectionInput`, etc.) for each tool, ensuring type safety and providing descriptions for the LLM.
*   **Custom Tool Classes:** Defines classes (`SearchFilingsTool`, `ExtractSectionTool`, etc.) inheriting from `langchain_core.tools.BaseTool`. Each class implements:
    *   `name`: Tool name for the LLM.
    *   `description`: Detailed description for the LLM on when and how to use the tool.
    *   `args_schema`: Links to the Pydantic input schema.
    *   `_run`: The core logic that executes when the tool is called, including API interactions, RAG logic (for `ExtractSectionTool`), error handling, and logging.
*   **LLM and Agent Setup:**
    *   Initializes the `ChatGoogleGenerativeAI` model.
    *   Creates the list of `tools`.
    *   Defines the `ChatPromptTemplate` containing the detailed system prompt (instructions, strategy, section identifiers, and final answer summary format with source URL inclusion) and placeholders for input and scratchpad.
    *   Creates the agent using `create_tool_calling_agent`.
    *   Creates the `AgentExecutor` to run the agent-tool loop.
*   **Main Execution Block (`if __name__ == "__main__":`)**:
    *   Checks for command-line arguments (`sys.argv`).
    *   If arguments exist, runs the `agent_executor.invoke` once with the combined arguments as input.
    *   If no arguments exist, enters the `while True` loop for interactive input.
    *   Handles user input, calls `agent_executor.invoke`, prints the final output, and includes basic error handling.

## 8. Logging Details

*   Logs are written to files in the `logs/` directory.
*   Filenames are timestamped in the format `log_YYYY-MM-DD_HH-MM-SS_ZONE.log` (e.g., `log_2025-03-26_20-06-53_PDT.log`).
*   Timestamps within the log file are in PST.
*   The default log level is `INFO`.
*   Logs include: Initialization steps, API key status, tool calls with parameters, API response summaries (e.g., number of filings found), RAG process steps (chunking, vector store creation, retrieval query, number/size of retrieved chunks), errors with tracebacks, and the agent's final output.
*   **Limitation:** The current file logging setup does *not* capture the LLM's internal "thought" process (the step-by-step reasoning printed to the console when `verbose=True` is used in `AgentExecutor`).

## 9. Future Enhancements / Roadmap Ideas

Based on development discussions, potential next steps include:

1.  **User-Facing Transparency:** Modify the system prompt to instruct the agent to include a brief summary of its key steps (e.g., identified ticker, filing used, section extracted) in its final answer to build user trust and allow for quick validation by domain experts.
2.  **Tool Usage Reliability:** Address the occasional issue where the LLM-based agent may incorrectly decide not to use available tools, instead generating placeholder responses. Potential solutions include prompt refinement, model switching (from Flash to Pro), or implementing a validation layer that confirms tools were actually used for data-dependent queries.
3.  **Detailed Internal Logging:** Implement a custom LangChain `BaseCallbackHandler` to capture the LLM's internal thoughts, detailed tool inputs/outputs, and other intermediate steps directly into the log files for more robust debugging and analysis by the developer.
4.  **Testing Framework:** Implement unit and integration tests using `pytest` to verify the functionality of individual tools and the agent's overall behavior.
5.  **Linting:** Integrate `flake8` into the development workflow to ensure code quality and consistency.
6.  **Advanced RAG (Optional):** Explore more sophisticated RAG techniques like HyDE (Hypothetical Document Embeddings) if the current simple RAG proves insufficient for complex queries (currently noted in `@suggestions.txt`).
7.  **Formal Roadmap:** Define specific goals and milestones in a `@roadmap.txt` file.

## 10. License

(Specify License Here - e.g., MIT License, Apache 2.0, or "Proprietary") 