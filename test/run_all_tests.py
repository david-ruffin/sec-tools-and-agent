#!/usr/bin/env python3
"""
Improved master script to run SEC agent tests sequentially with proper error handling
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

# Define simplified test suites
TEST_SUITES = [
    {
        "name": "Basic Functionality Test",
        "description": "Tests basic functionality of the SEC agent",
        "questions": [
            "What is the ticker symbol for Apple Inc?",
            "When was the latest 10-K filed by Microsoft?"
        ]
    },
    {
        "name": "RAG Processing Test",
        "description": "Tests RAG processing for different section types",
        "questions": [
            "Summarize Apple's business model from their latest 10-K",
            "What are Microsoft's biggest risks in their latest 10-K?"
        ]
    },
    {
        "name": "SEC API Feature Test",
        "description": "Tests specific SEC API features",
        "questions": [
            "What was Tesla's revenue in 2023?",
            "How many shares does Tim Cook own of Apple?"
        ]
    }
]

def run_single_query(question):
    """Run a single query and capture all output and metrics"""
    print(f"\n{'-'*80}")
    print(f"EXECUTING: {question}")
    print(f"{'-'*80}")
    
    cmd = [sys.executable, "sec.py", "--query", question]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise an exception on non-zero exit
        )
        
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
        
        # Print a summary of the output
        print(f"Exit code: {exit_code}")
        if stdout:
            # Print just the first few lines and last few lines
            stdout_lines = stdout.splitlines()
            if len(stdout_lines) > 20:
                print("First 10 lines of output:")
                for line in stdout_lines[:10]:
                    print(f"  {line}")
                print("...")
                print("Last 10 lines of output:")
                for line in stdout_lines[-10:]:
                    print(f"  {line}")
            else:
                print("Output:")
                for line in stdout_lines:
                    print(f"  {line}")
        
        if stderr:
            print("Error output:")
            for line in stderr.splitlines():
                print(f"  {line}")
        
        # Determine success/failure
        if exit_code == 0:
            status = "SUCCESS"
        else:
            status = "FAILURE"
            
        # Check if RAG was used
        rag_used = "Processing text with RAG" in stdout
        
    except Exception as e:
        stdout = ""
        stderr = str(e)
        exit_code = -1
        status = "ERROR"
        rag_used = False
    
    elapsed_time = time.time() - start_time
    
    print(f"Status: {status}")
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"RAG used: {rag_used}")
    
    return {
        "question": question,
        "status": status,
        "exit_code": exit_code,
        "time": elapsed_time,
        "rag_used": rag_used,
        "stdout": stdout,
        "stderr": stderr
    }

def run_test_suite(suite):
    """Run all questions in a test suite"""
    suite_results = {
        "name": suite["name"],
        "description": suite["description"],
        "start_time": datetime.now().isoformat(),
        "results": []
    }
    
    print(f"\n{'='*80}")
    print(f"RUNNING TEST SUITE: {suite['name']}")
    print(f"Description: {suite['description']}")
    print(f"Questions: {len(suite['questions'])}")
    print(f"{'='*80}")
    
    for i, question in enumerate(suite["questions"], 1):
        print(f"\nQuestion {i}/{len(suite['questions'])}")
        result = run_single_query(question)
        suite_results["results"].append(result)
        
        # Add a delay between questions
        if i < len(suite["questions"]):
            delay = 5
            print(f"Waiting {delay} seconds before next question...")
            time.sleep(delay)
    
    # Calculate suite summary
    success_count = sum(1 for r in suite_results["results"] if r["status"] == "SUCCESS")
    rag_count = sum(1 for r in suite_results["results"] if r["rag_used"])
    
    suite_results["end_time"] = datetime.now().isoformat()
    suite_results["total_questions"] = len(suite["questions"])
    suite_results["success_count"] = success_count
    suite_results["rag_count"] = rag_count
    
    # Print summary
    print(f"\n{'-'*80}")
    print(f"SUITE SUMMARY: {suite['name']}")
    print(f"Total questions: {len(suite['questions'])}")
    print(f"Successful: {success_count}")
    print(f"RAG used: {rag_count}")
    print(f"{'-'*80}")
    
    return suite_results

def main():
    """Run all test suites and generate a report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.json"
    
    print(f"Starting SEC Agent test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {len(TEST_SUITES)} test suites with a total of {sum(len(s['questions']) for s in TEST_SUITES)} questions")
    print(f"Results will be saved to {results_file}")
    
    all_results = {
        "timestamp": timestamp,
        "suites": []
    }
    
    for i, suite in enumerate(TEST_SUITES, 1):
        print(f"\nRunning suite {i}/{len(TEST_SUITES)}")
        suite_results = run_test_suite(suite)
        all_results["suites"].append(suite_results)
        
        # Add a delay between suites
        if i < len(TEST_SUITES):
            delay = 10
            print(f"\nWaiting {delay} seconds before next suite...")
            time.sleep(delay)
    
    # Calculate overall summary
    total_questions = sum(len(s["questions"]) for s in TEST_SUITES)
    successful_questions = sum(suite["success_count"] for suite in all_results["suites"])
    rag_questions = sum(suite["rag_count"] for suite in all_results["suites"])
    
    all_results["total_questions"] = total_questions
    all_results["successful_questions"] = successful_questions
    all_results["rag_questions"] = rag_questions
    
    # Print overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total test suites: {len(TEST_SUITES)}")
    print(f"Total questions: {total_questions}")
    print(f"Successful questions: {successful_questions}")
    print(f"Questions using RAG: {rag_questions}")
    
    # Save detailed results to file
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")

if __name__ == "__main__":
    main() 