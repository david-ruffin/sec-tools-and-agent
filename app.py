#!/usr/bin/env python3
"""
SEC Filing Analysis System - Web Interface
Flask-based web server that provides API endpoints for the React frontend
"""

import os
import json
import datetime
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template, Response
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import csv
import io

# Azure Storage imports
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError

# Import our core functionality from existing modules
from main import SecFilingDownloader, SEC_API_KEY, OPENAI_API_KEY
from sec_analyzer import analyze_sec_filing

# Load environment variables - works for both local .env and Azure App Settings
load_dotenv()  # This loads from .env file if it exists, but won't fail if the file is missing in Azure

# Get Azure Storage connection string
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Configure logging
log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"sec_web_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a filter to add trace_id to all log records
class TraceIDFilter(logging.Filter):
    def __init__(self, name=''):
        super().__init__(name)
        self.trace_id = "no_trace"
        
    def filter(self, record):
        record.trace_id = self.trace_id
        return True

# Create and add the filter
trace_filter = TraceIDFilter()

# Configure handlers with the filter
file_handler = logging.FileHandler(log_file)
file_handler.addFilter(trace_filter)
console_handler = logging.StreamHandler()
console_handler.addFilter(trace_filter)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(trace_id)s',
    handlers=[
        file_handler,
        console_handler
    ]
)
logger = logging.getLogger('sec_web')

# Add trace_id to log records
class TraceLogger:
    def __init__(self, logger, trace_filter):
        self.logger = logger
        self.trace_filter = trace_filter
    
    def set_trace_id(self, trace_id):
        self.trace_filter.trace_id = trace_id
        return self
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

traced_logger = TraceLogger(logger, trace_filter)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize our SEC filing downloader
downloader = SecFilingDownloader(SEC_API_KEY)

# Initialize the Azure Blob Storage client
blob_service_client = None
if AZURE_STORAGE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        logger.info("Azure Blob Storage client initialized")
        
        # Ensure the 'data' container exists
        try:
            container_client = blob_service_client.get_container_client("data")
            container_client.get_container_properties()  # Will raise if container doesn't exist
            logger.info("Azure 'data' container exists")
        except ResourceNotFoundError:
            container_client = blob_service_client.create_container("data")
            logger.info("Created Azure 'data' container")
    except Exception as e:
        logger.error(f"Failed to initialize Azure Blob Storage: {str(e)}")
        blob_service_client = None
else:
    logger.warning("AZURE_STORAGE_CONNECTION_STRING not set, using local storage")

# File paths for local storage fallback
TEST_RESULTS_FILE = "data/test_results.json"
FINE_TUNING_FILE = "data/fine_tuning_data.jsonl"

# Ensure local data directory exists for fallback
os.makedirs(os.path.dirname(TEST_RESULTS_FILE), exist_ok=True)

# Store test results in memory with persistence
test_results = []

# Load test results from Azure Blob Storage or local file
def load_test_results():
    global test_results
    
    # Try to load from Azure Blob Storage first if configured
    if blob_service_client:
        try:
            # Get a blob client for the test results file
            blob_client = blob_service_client.get_blob_client(
                container="data", 
                blob="test_results.json"
            )
            
            # Download and parse the blob
            download_stream = blob_client.download_blob()
            file_content = download_stream.readall()
            
            if file_content:
                test_results = json.loads(file_content)
                logger.info(f"Successfully loaded {len(test_results)} test results from Azure Blob Storage")
                return test_results
        except ResourceNotFoundError:
            logger.info("Test results file not found in Azure Blob Storage, will create a new one")
        except Exception as e:
            logger.error(f"Error loading test results from Azure: {str(e)}")
    
    # Fall back to local file if Azure fails or is not configured
    if os.path.exists(TEST_RESULTS_FILE):
        try:
            with open(TEST_RESULTS_FILE, "r") as f:
                test_results = json.load(f)
                logger.info(f"Successfully loaded {len(test_results)} test results from local file")
                return test_results
        except Exception as e:
            logger.error(f"Error loading test results from local file: {str(e)}")
    
    # If all else fails, return an empty list
    logger.warning("No test results found, starting with empty list")
    return []

# Save test results to Azure Blob Storage or local file
def save_test_results():
    # Convert test results to JSON string
    test_results_json = json.dumps(test_results, indent=2)
    
    # Try to save to Azure Blob Storage first if configured
    if blob_service_client:
        try:
            # Get a blob client for the test results file
            blob_client = blob_service_client.get_blob_client(
                container="data", 
                blob="test_results.json"
            )
            
            # Upload the JSON string to the blob
            blob_client.upload_blob(
                test_results_json,
                content_settings=ContentSettings(content_type="application/json"),
                overwrite=True
            )
            
            logger.info(f"Successfully saved {len(test_results)} test results to Azure Blob Storage")
            return True
        except Exception as e:
            logger.error(f"Error saving test results to Azure: {str(e)}")
    
    # Fall back to local file if Azure fails or is not configured
    try:
        with open(TEST_RESULTS_FILE, "w") as f:
            f.write(test_results_json)
            f.flush()  # Force flush to disk
        logger.info(f"Successfully saved {len(test_results)} test results to local file")
        return True
    except Exception as e:
        logger.error(f"Error saving test results to local file: {str(e)}")
        return False

# Update fine-tuning data from test results
def update_fine_tuning_data():
    # Format specifically for ML fine-tuning or RAG
    fine_tuning_data = []
    for result in test_results:
        if result['status'] == 'success':
            # Only include successful results
            ft_entry = {
                "query": result.get('query', ''),
                "response": result.get('analysis', ''),
                "context": result.get('context', ''),
                "metadata": {
                    "company": result.get('company', ''),
                    "formType": result.get('formType', ''),
                    "year": result.get('year', ''),
                    "timestamp": result.get('timestamp', '')
                }
            }
            
            # Add feedback if available
            if 'feedback' in result:
                ft_entry["feedback"] = result['feedback']
            if 'rating' in result:
                ft_entry["rating"] = result['rating']
                
            # Add human feedback flag if rating and feedback exist - useful for RLHF
            if 'rating' in result and 'feedback' in result and result['feedback'].strip():
                ft_entry["has_human_feedback"] = True
                
            fine_tuning_data.append(ft_entry)
    
    # Convert to JSONL (each line is a valid JSON object)
    jsonl_output = '\n'.join([json.dumps(entry) for entry in fine_tuning_data])
    
    # Try to save to Azure Blob Storage first if configured
    if blob_service_client:
        try:
            # Get a blob client for the fine tuning data file
            blob_client = blob_service_client.get_blob_client(
                container="data", 
                blob="fine_tuning_data.jsonl"
            )
            
            # Upload the JSONL string to the blob
            blob_client.upload_blob(
                jsonl_output,
                content_settings=ContentSettings(content_type="application/x-jsonlines"),
                overwrite=True
            )
            
            logger.info(f"Updated fine tuning data in Azure Blob Storage with {len(fine_tuning_data)} entries")
            return fine_tuning_data
        except Exception as e:
            logger.error(f"Error saving fine tuning data to Azure: {str(e)}")
    
    # Fall back to local file if Azure fails or is not configured
    try:
        with open(FINE_TUNING_FILE, 'w') as f:
            f.write(jsonl_output)
            logger.info(f"Updated fine tuning data in local file with {len(fine_tuning_data)} entries")
        return fine_tuning_data
    except Exception as e:
        logger.error(f"Error saving fine tuning data to local file: {str(e)}")
        return fine_tuning_data

# Initialize with saved results
test_results = load_test_results()

# Current log file for reference in test results
current_log_file = log_file

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('modern-index.html')

@app.route('/legacy')
def legacy_index():
    """Serve the legacy version of the application."""
    return render_template('index.html')

@app.route('/api/company/search', methods=['GET'])
def search_company():
    """Search for a company by name or ticker."""
    name = request.args.get('query', '')
    if not name:
        return jsonify({"error": "No company name provided"}), 400
    
    # Use the existing company lookup functionality
    company_info = downloader.company_lookup.find_company_by_name(name)
    if not company_info:
        return jsonify({"error": f"Could not find company: {name}"}), 404
    
    return jsonify({
        "company": company_info['title'],
        "cik": company_info['cik_str'],
        "ticker": company_info['ticker']
    })

@app.route('/api/filing/analyze', methods=['POST'])
def analyze_filing():
    """Process a query about an SEC filing."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Extract parameters from the request
    query = data.get('query')
    company = data.get('company')
    form_type = data.get('formType', '10-K')
    year = data.get('year')
    test_id = data.get('testId')
    
    # Generate a test ID if not provided
    if not test_id:
        test_id = f"test-{int(datetime.datetime.now().timestamp())}"
    
    if not query or not company or not year:
        return jsonify({
            "error": "Missing required parameters",
            "required": ["query", "company", "year"]
        }), 400
    
    try:
        # Use the existing functionality to process the query
        result = downloader.process_query_with_filing(
            query=query,
            company_name=company,
            form_type=form_type,
            year=year
        )
        
        if isinstance(result, dict):
            # Make the PDF path relative for the frontend
            pdf_path = result['pdf_path']
            
            # If pdf_path is a full URL (from Azure Blob Storage), use it directly
            if pdf_path.startswith('http'):
                relative_path = pdf_path
            else:
                # Otherwise, it's a local file path, so extract just the filename
                relative_path = f"/api/filings/{os.path.basename(pdf_path)}"
            
            response_data = {
                "analysis": result['analysis'],
                "pdfPath": relative_path,
                "id": test_id  # Include the test ID in the response
            }
            
            # Save this as a test result if it's part of a test
            if test_id:
                # Generate a trace ID for log correlation
                trace_id = str(uuid.uuid4())
                traced_logger.set_trace_id(trace_id).info(f"Test result recorded for query: {query}")
                
                # Extract the context if available (useful for fine-tuning)
                # This will be the sections from the filing that were used for the answer
                context = result.get('context', '')
                if not context and 'filing_text' in result:
                    # If there's no explicit context but we have the filing text, use that
                    context = result.get('filing_text', '')[:1000]  # Limit to first 1000 chars
                
                test_result = {
                    "id": test_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query": query,  # Original question
                    "company": company,
                    "formType": form_type,
                    "year": year,
                    "analysis": result['analysis'],
                    "pdfPath": relative_path,
                    "status": "success",
                    "log_file": current_log_file,
                    "trace_id": trace_id,
                    "context": context  # For fine-tuning/RAG
                }
                test_results.append(test_result)
            
            # Save updated test results
            save_test_results()
            
            return jsonify(response_data)
        else:
            # Error case
            error_response = {"error": str(result)}
            
            # Save this as a test result if it's part of a test
            if test_id:
                # Generate a trace ID for log correlation
                trace_id = str(uuid.uuid4())
                traced_logger.set_trace_id(trace_id).error(f"Error in test for query: {query}. Error: {str(result)}")
                
                test_result = {
                    "id": test_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query": query,  # Original question
                    "company": company,
                    "formType": form_type,
                    "year": year,
                    "error": str(result),
                    "status": "error",
                    "log_file": current_log_file,
                    "trace_id": trace_id
                }
                test_results.append(test_result)
            
            # Save updated test results
            save_test_results()
                
            return jsonify(error_response), 500
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_response = {"error": f"Error processing query: {str(e)}"}
        
        # Save this as a test result if it's part of a test
        if test_id:
            # Generate a trace ID for log correlation
            trace_id = str(uuid.uuid4())
            traced_logger.set_trace_id(trace_id).error(f"Exception in test for query: {query}. Error: {str(e)}")
            
            test_result = {
                "id": test_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query,  # Original question
                "company": company,
                "formType": form_type,
                "year": year,
                "error": str(e),
                "status": "error",
                "log_file": current_log_file,
                "trace_id": trace_id
            }
            test_results.append(test_result)
            
            # Save updated test results
            save_test_results()
            
        return jsonify(error_response), 500

@app.route('/api/filing/extract', methods=['POST'])
def extract_parameters():
    """Extract parameters from a natural language query."""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query']
    trace_id = str(uuid.uuid4())
    traced_logger.set_trace_id(trace_id).info(f"Parameter extraction request for query: {query}")
    
    # Use the existing parameter extraction functionality
    params = downloader.extract_parameters(query)
    
    # Handle 'latest' queries
    current_year = datetime.datetime.now().year
    if 'latest' in query.lower():
        traced_logger.info("Query contains 'latest', setting year to current year")
        if not params['year']:
            params['year'] = str(current_year)
    
    # Provide fallbacks for missing parameters
    if not params['form_type']:
        params['form_type'] = '10-K'  # Default to 10-K if not specified
    
    # If we still don't have a year but have a company, use current year
    if params['company'] and not params['year']:
        params['year'] = str(current_year)
    
    traced_logger.info(f"Extracted parameters: {json.dumps(params)}")
    
    return jsonify({
        "company": params['company'],
        "formType": params['form_type'],
        "year": params['year'],
        "query": query,
        "complete": bool(params['company'])  # Consider complete if at least company is identified
    })

@app.route('/api/filings/<path:filename>')
def serve_filing(filename):
    """Serve a filing PDF file from Azure Storage."""
    import io
    from flask import send_file
    
    storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    storage_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "filings")
    
    try:
        # Initialize Azure client
        from azure.storage.blob import BlobServiceClient
        
        # Get the blob client
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client(storage_container_name)
        blob_client = container_client.get_blob_client(filename)
        
        # Download the blob content
        data = blob_client.download_blob().readall()
        
        # Create a file-like object from the data
        file_like_object = io.BytesIO(data)
        
        # Send the file directly
        return send_file(
            file_like_object,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return jsonify({"error": "Could not access file"}), 500

@app.route('/api/feedback', methods=['POST'])
def collect_feedback():
    """Collect user feedback on analysis quality."""
    data = request.json
    if not data:
        return jsonify({"error": "No feedback provided"}), 400
    
    # We would normally store this in a database, but for now just log it
    logger.info(f"Feedback received: {json.dumps(data)}")
    
    # If this feedback is associated with a test, add it to the test result
    test_id = data.get('testId')
    if test_id:
        for result in test_results:
            if result['id'] == test_id:
                result['feedback'] = data.get('feedback')
                result['rating'] = data.get('rating')
                break
    
    # Save updated test results
    save_test_results()
    
    return jsonify({"status": "Feedback received, thank you!"})

@app.route('/api/test-results', methods=['GET'])
def get_test_results():
    """Get all test results."""
    return jsonify(test_results)

@app.route('/api/test-results/<test_id>', methods=['GET'])
def get_test_result(test_id):
    """Get a specific test result."""
    for result in test_results:
        if result['id'] == test_id:
            return jsonify(result)
    return jsonify({"error": "Test result not found"}), 404

@app.route('/api/test-results/<test_id>/feedback', methods=['POST'])
def submit_test_feedback(test_id):
    """Submit feedback for a specific test result."""
    data = request.json
    if not data:
        return jsonify({"error": "No feedback data provided"}), 400
    
    feedback = data.get('feedback', '')
    rating = data.get('rating')
    
    # Find the test result to update
    found = False
    for result in test_results:
        if result['id'] == test_id:
            # Add feedback and rating to the test result
            result['feedback'] = feedback
            if rating is not None:
                result['rating'] = rating
            
            # Log the feedback submission with the same trace ID
            trace_id = result.get('trace_id', str(uuid.uuid4()))
            traced_logger.set_trace_id(trace_id).info(f"Feedback submitted for test {test_id}: Rating={rating}, Feedback={feedback}")
            
            # Save updated test results
            save_test_results()
            
            # Update fine_tuning_data.jsonl after feedback submission
            update_fine_tuning_data()
            
            found = True
            break
    
    # If test result not found, create a new one with this feedback
    if not found:
        # Create a new test result entry with the feedback
        new_result = {
            'id': test_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'feedback_only',
            'feedback': feedback,
            'trace_id': str(uuid.uuid4())
        }
        
        if rating is not None:
            new_result['rating'] = rating
            
        # Add dynamic fields from the test ID if it follows our naming pattern
        if '-' in test_id:
            parts = test_id.split('-', 1)
            new_result['source'] = parts[0]
        
        test_results.append(new_result)
        traced_logger.set_trace_id(new_result['trace_id']).info(f"Created new feedback-only entry with ID {test_id}")
        
        # Save updated test results
        save_test_results()
        
        # Update fine_tuning_data.jsonl after feedback submission
        update_fine_tuning_data()
    
    return jsonify({"success": True, "message": "Feedback submitted successfully"})

@app.route('/api/test-results/export', methods=['GET'])
def export_test_results():
    """Export test results in various formats."""
    format_type = request.args.get('format', 'csv')
    
    if format_type == 'json':
        return jsonify(test_results)
    
    elif format_type == 'jsonl' or format_type == 'fine-tuning':
        # Use the function to generate fine-tuning data
        fine_tuning_data = update_fine_tuning_data()
        
        # Return the data directly like the JSON export, but with the proper filename
        response = jsonify(fine_tuning_data)
        response.headers['Content-Disposition'] = 'attachment;filename=fine_tuning_data.jsonl'
        return response
    
    # Default to CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    if test_results:
        writer.writerow(test_results[0].keys())
        
        # Write data rows
        for result in test_results:
            writer.writerow(result.values())
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=test_results.csv"}
    )

@app.route('/test-results')
def test_results_page():
    """Display the test results page."""
    # Check if we have test_results.html template file and use it
    if os.path.exists('templates/test_results.html'):
        return render_template('test_results.html')
    # Otherwise generate it on the fly
    else:
        return """
<!DOCTYPE html>
<html>
<head>
    <title>SEC Filing Analyzer - Test Results</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #033c73; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .actions {{ margin-top: 20px; }}
        .actions a {{ display: inline-block; margin-right: 10px; padding: 8px 15px; background-color: #033c73; color: white; text-decoration: none; border-radius: 4px; }}
        .actions a:hover {{ background-color: #022954; }}
        .filter-controls {{ margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
        .filter-controls label {{ margin-right: 15px; }}
        .filter-controls select, .filter-controls input {{ padding: 5px; }}
        
        /* Modal styles */
        .modal {{ display: none; position: fixed; z-index: 1; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.4); }}
        .modal-content {{ background-color: #fefefe; margin: 10% auto; padding: 20px; border: 1px solid #888; width: 50%; border-radius: 5px; }}
        .close {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }}
        .close:hover {{ color: black; }}
        .star-rating {{ margin: 10px 0; }}
        .star {{ color: #ddd; cursor: pointer; font-size: 24px; }}
        .star.active {{ color: gold; }}
        textarea {{ width: 100%; height: 100px; margin: 10px 0; padding: 8px; box-sizing: border-box; }}
        .submit-btn {{ background-color: #033c73; color: white; border: none; padding: 8px 15px; cursor: pointer; border-radius: 4px; }}
        .submit-btn:hover {{ background-color: #022954; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SEC Filing Analyzer - Test Results</h1>
        <p>View and analyze test results from the SEC Filing Analyzer.</p>
        
        <div class="filter-controls">
            <label>Status: 
                <select id="statusFilter">
                    <option value="all">All</option>
                    <option value="success">Success</option>
                    <option value="error">Error</option>
                </select>
            </label>
            <label>Company: 
                <input type="text" id="companyFilter" placeholder="Filter by company...">  
            </label>
            <button id="applyFilters">Apply Filters</button>
        </div>
        
        <div class="actions">
            <a href="/api/test-results/export?format=csv" download>Export as CSV</a>
            <a href="/api/test-results/export?format=json" download>Export as JSON</a>
            <a href="/api/test-results/export?format=jsonl" download>Export for Fine-tuning (JSONL)</a>
            <a href="/">Back to Main App</a>
        </div>
        
        <table id="resultsTable">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Timestamp</th>
                    <th>Company</th>
                    <th>Form Type</th>
                    <th>Year</th>
                    <th>Query</th>
                    <th>Status</th>
                    <th>Rating</th>
                    <th>Feedback</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody id="resultsBody">
                <!-- Results will be loaded here via JavaScript -->
            </tbody>
        </table>
    </div>
    
    <!-- Feedback Modal -->
    <div id="feedbackModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeFeedbackModal()">&times;</span>
            <h2>Submit Feedback</h2>
            <p id="feedbackInfo">Providing feedback on test result</p>
            
            <div class="rating-options">
                <span>How accurate was the analysis?</span>
                <div class="rating-option">
                    <input type="radio" id="rating1" name="rating" value="1">
                    <label for="rating1">
                        <strong>⭐ Incorrect</strong> - The analysis is factually wrong or misses the point
                    </label>
                </div>
                <div class="rating-option">
                    <input type="radio" id="rating2" name="rating" value="2">
                    <label for="rating2">
                        <strong>⭐⭐ Partially Correct</strong> - The analysis has some valid points but contains errors
                    </label>
                </div>
                <div class="rating-option">
                    <input type="radio" id="rating3" name="rating" value="3">
                    <label for="rating3">
                        <strong>⭐⭐⭐ Spot On</strong> - The analysis correctly addresses the question
                    </label>
                </div>
            </div>
            
            <div>
                <p>Additional comments:</p>
                <textarea id="feedbackText" placeholder="Please provide any additional feedback on the analysis result..."></textarea>
            </div>
            <input type="hidden" id="testId">
            <button class="submit-btn" onclick="submitFeedback()">Submit Feedback</button>
        </div>
    </div>

    <script>
        // Feedback modal functions
        let currentRating = 0;
        
        function showFeedbackModal(testId, company, query, rating, feedback) {
            document.getElementById('feedbackModal').style.display = 'block';
            document.getElementById('feedbackInfo').textContent = `Providing feedback for query: "${query}" for company: ${company}`;
            document.getElementById('testId').value = testId;
            document.getElementById('feedbackText').value = feedback || '';
            
            // Reset stars
            document.querySelectorAll('.star').forEach(star => {
                star.classList.remove('active');
            });
            
            // Set rating if exists
            if (rating) {
                setRating(parseInt(rating));
            }
        }
        
        function closeFeedbackModal() {
            document.getElementById('feedbackModal').style.display = 'none';
        }
        
        function setRating(rating) {
            currentRating = rating;
        }
        
        function submitFeedback() {
            const testId = document.getElementById('testId').value;
            const feedback = document.getElementById('feedbackText').value;
            
            // Get the selected rating (this is the fix from legacy)
            let rating = null;
            document.querySelectorAll('input[name="rating"]').forEach(radio => {
                if (radio.checked) {
                    rating = parseInt(radio.value);
                }
            });
            
            // Submit the feedback without validation
            fetch(`/api/test-results/${testId}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    feedback: feedback,
                    rating: rating
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Close modal and show success
                    document.getElementById('feedbackModal').style.display = 'none';
                    loadResults(); // Refresh the results
                } else {
                    alert('Error submitting feedback: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                alert('Error submitting feedback: ' + error);
            });
        }
        
        // Fetch and display test results
        // Load results function
        function loadResults() {{
            console.log('Loading test results...');
            
        // When document is ready
        document.addEventListener('DOMContentLoaded', function() {{                
                const statusFilter = document.getElementById('statusFilter').value;
                const companyFilter = document.getElementById('companyFilter').value.toLowerCase();
                
                fetch('/api/test-results')
                    .then(response => response.json())
                    .then(data => {{                        
                        const resultsBody = document.getElementById('resultsBody');
                        resultsBody.innerHTML = '';
                        
                        // Apply filters
                        const filteredData = data.filter(result => {{
                            // Handle potential missing fields (for robust filtering)
                            if (statusFilter !== 'all' && result.status !== statusFilter) return false;
                            
                            // Safe company filtering - handle missing company field
                            if (companyFilter && (!result.company || !result.company.toLowerCase().includes(companyFilter))) return false;
                            
                            return true;
                        }});
                        
                        // Sort by timestamp (newest first)
                        // Sort by timestamp (newest first)
                        filteredData.sort((a, b) => {
                            // Handle missing timestamps
                            if (!a.timestamp) return 1;
                            if (!b.timestamp) return -1;
                            return new Date(b.timestamp) - new Date(a.timestamp);
                        });
                        
                        // Debug - check if we have any results
                        console.log(`Displaying ${filteredData.length} results after filtering`);
                        
                        filteredData.forEach(result => {{                            
                            const row = document.createElement('tr');
                            
                            // Format rating display
                            let ratingDisplay = 'Not rated';
                            if (result.rating !== undefined) {
                                if (result.rating === 1) {
                                    ratingDisplay = '⭐ Incorrect';
                                } else if (result.rating === 2) {
                                    ratingDisplay = '⭐⭐ Partially Correct';
                                } else if (result.rating === 3) {
                                    ratingDisplay = '⭐⭐⭐ Spot On';
                                } else {
                                    ratingDisplay = `Rating: ${result.rating}`;
                                }
                            }
                            
                            // Handle missing fields safely
                            const company = result.company || 'N/A';
                            const formType = result.formType || 'N/A';
                            const year = result.year || 'N/A';
                            const query = result.query || 'N/A';
                            const status = result.status || 'unknown';
                            const timestamp = result.timestamp ? new Date(result.timestamp).toLocaleString() : 'N/A';
                            
                            // Truncate feedback for display
                            let feedbackDisplay = 'No feedback';
                            if (result.feedback) {
                                feedbackDisplay = result.feedback.length > 50 ? 
                                    result.feedback.substring(0, 50) + '...' : 
                                    result.feedback;
                            }
                            
                            row.innerHTML = `
                                <td>${result.id || 'Unknown'}</td>
                                <td>${timestamp}</td>
                                <td>${company}</td>
                                <td>${formType}</td>
                                <td>${year}</td>
                                <td>${query}</td>
                                <td class="${status}">${status}</td>
                                <td>${ratingDisplay}</td>
                                <td title="${result.feedback || ''}">${feedbackDisplay}</td>
                                <td>
                                    <a href="#" onclick="showFeedbackModal(\`${result.id}\`, \`${company}\`, \`${query}\`, \`${result.rating || 0}\`, \`${result.feedback || ''}\`); return false;">${result.feedback ? 'Edit' : 'Add'} Feedback</a>
                                    <a href="/api/test-results/${result.id}" target="_blank">View Details</a>
                                    ${result.pdfPath ? `<a href="${result.pdfPath}" target="_blank">View PDF</a>` : ''}
                                </td>
                            `;
                            resultsBody.appendChild(row);
                        }});
                    }});
            }}
            
            console.log('DOM loaded, initializing test results page');
            
            // Load results on page load
            loadResults();
            
            // Set up filter handlers
            document.getElementById('applyFilters').addEventListener('click', loadResults);
            
            // Add error handling for fetch
            window.addEventListener('error', function(e) {
                console.error('Global error:', e.message);
            });
        });
    </script>
</body>
</html>
        """
    return render_template('test_results.html')

if __name__ == "__main__":
    # Ensure the filings directory exists
    os.makedirs('filings', exist_ok=True)
    
    # Create a simple HTML template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        os.makedirs('templates', exist_ok=True)
        with open('templates/index.html', 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>SEC Filing Analyzer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        p { color: #666; }
        .note { background-color: #f8f9fa; padding: 15px; border-radius: 4px; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SEC Filing Analyzer API</h1>
        <p>This is the backend API for the SEC Filing Analyzer. The API endpoints are available at:</p>
        <ul>
            <li><code>/api/company/search?query=COMPANY_NAME</code> - Search for a company</li>
            <li><code>/api/filing/analyze</code> - Analyze a filing (POST)</li>
            <li><code>/api/filing/extract</code> - Extract parameters from a query (POST)</li>
            <li><code>/api/filings/FILENAME</code> - Serve a filing PDF</li>
            <li><code>/api/feedback</code> - Collect user feedback (POST)</li>
        </ul>
        <div class="note">
            <p>To use the full application, you need to connect the React frontend. This is just the API server.</p>
        </div>
    </div>
</body>
</html>
            """)
    
    # Get port from environment variable (for Azure App Service compatibility)
    port = int(os.environ.get('PORT', 8080))
    
    # In production, set debug to False
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)