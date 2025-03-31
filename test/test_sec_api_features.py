#!/usr/bin/env python3
"""
Test script for SEC API Features
This script tests various SEC API capabilities through the agent
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

# Test questions that exercise different SEC API features
SEC_API_TEST_QUESTIONS = [
    # Filing search with different form types
    "Find the most recent 8-K filings for Tesla.",
    
    # Filing search with date ranges
    "What 10-Q filings did Apple submit in 2023?",
    
    # Section extraction for different section types
    "Extract the Business Description (Section 1) from Amazon's latest 10-K.",
    
    # Section extraction for less common sections
    "What does Intel's latest 10-K say about their properties (Section 2)?",
    
    # XBRL data extraction for Income Statement
    "Show me NVIDIA's Income Statement from their most recent annual report.",
    
    # XBRL data extraction for Balance Sheet
    "Extract the Balance Sheet from Facebook's most recent 10-K.",
    
    # XBRL data with specific metrics
    "What were Google's total assets and liabilities in their latest 10-K?",
    
    # Multiple company comparison using SEC API
    "Compare the revenue growth rates of Microsoft, Apple, and Amazon from their latest annual reports.",
    
    # Industry-specific query
    "Find healthcare companies that mentioned COVID-19 in their latest 10-K risk factors.",
    
    # Query using CIK instead of ticker
    "Show me the latest filings for CIK 0000320193."
]

def run_api_test(question):
    """Run a single test question through the SEC agent and log results"""
    print(f"\n{'='*80}")
    print(f"TESTING SEC API: {question}")
    print(f"{'='*80}")
    
    # Construct the command to run the SEC agent with the question
    cmd = [sys.executable, "sec.py", question]
    
    # Record start time
    start_time = time.time()
    
    # Run the command and capture output
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        print(f"Output sample (first 300 chars):\n{output[:300]}...")
        status = "SUCCESS"
    except subprocess.CalledProcessError as e:
        output = f"{e.stdout}\n{e.stderr}"
        print(f"ERROR (exit code {e.returncode}):\n{output[:300]}...")
        status = "ERROR"
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Detect which APIs were used
    api_usage = {
        "filing_search": "sec_filing_search" in output,
        "section_extraction": "sec_section_extractor" in output,
        "xbrl_extraction": "sec_xbrl_extractor" in output,
        "rag_processing": "Processing text with RAG" in output
    }
    
    result = {
        "question": question,
        "status": status,
        "time": elapsed_time,
        "api_usage": api_usage
    }
    
    # Print API usage
    print("\nAPI Usage:")
    for api, used in api_usage.items():
        print(f"  {api}: {'✓' if used else '✗'}")
    
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Status: {status}")
    
    return result

def main():
    """Run all SEC API test questions and summarize results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"sec_api_test_results_{timestamp}.json"
    
    print(f"Starting SEC API feature test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing {len(SEC_API_TEST_QUESTIONS)} questions")
    print(f"Results will be saved to {results_file}")
    
    results = []
    
    for i, question in enumerate(SEC_API_TEST_QUESTIONS, 1):
        print(f"\nTest {i}/{len(SEC_API_TEST_QUESTIONS)}")
        result = run_api_test(question)
        results.append(result)
        
        # Add a delay between tests to avoid rate limiting
        if i < len(SEC_API_TEST_QUESTIONS):
            print("\nWaiting 5 seconds before next test...")
            time.sleep(5)
    
    # Summarize results
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    
    # Count API usage across all tests
    api_usage_counts = {
        api: sum(1 for r in results if r["api_usage"][api])
        for api in results[0]["api_usage"].keys()
    }
    
    print(f"\n{'='*80}")
    print(f"SEC API TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    print("\nAPI Usage Statistics:")
    for api, count in api_usage_counts.items():
        print(f"  {api}: {count}/{len(results)} ({count/len(results):.0%})")
    
    # Save detailed results to file
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "questions": len(results),
            "success_count": success_count,
            "api_usage_counts": api_usage_counts,
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")

if __name__ == "__main__":
    main() 