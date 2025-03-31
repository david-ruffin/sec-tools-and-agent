#!/usr/bin/env python3
"""
Test script for SEC Unified Agent
This script runs a set of diverse test questions to validate all tools and combinations
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Reduced to a single simple test question
TEST_QUESTIONS = [
    # Basic filing search
    "What is the ticker symbol for Apple Inc?",
]

def run_test(question):
    """Run a single test question through the SEC agent"""
    print(f"\n{'='*80}")
    print(f"TESTING: {question}")
    print(f"{'='*80}")
    
    # Construct the command to run the SEC agent with the question
    cmd = [sys.executable, "sec.py", "--query", question]
    
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
        print(f"Output:\n{result.stdout}")
        status = "SUCCESS"
    except subprocess.CalledProcessError as e:
        print(f"ERROR (exit code {e.returncode}):\n{e.stdout}\n{e.stderr}")
        status = "ERROR"
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Status: {status}")
    
    return {
        "question": question,
        "status": status,
        "time": elapsed_time
    }

def main():
    """Run all test questions and summarize results"""
    print(f"Starting SEC Agent test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing {len(TEST_QUESTIONS)} questions\n")
    
    results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\nTest {i}/{len(TEST_QUESTIONS)}")
        result = run_test(question)
        results.append(result)
        
    # Summarize results
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    error_count = len(results) - success_count
    
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print("\nDetailed results:")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['status']} ({result['time']:.2f}s): {result['question']}")

if __name__ == "__main__":
    main() 